"""Bộ chấm tự động Tuần 5 — phần Tầng 1 (bộ đồ nghề học sâu).

    python3 -m pytest bai_tap/test_all.py -q            # chấm bài CỦA BẠN
    SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q # chấm ĐÁP ÁN
"""
import os, importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

SRC = Path(__file__).parent.parent / ("dap_an" if os.environ.get("SOLUTION") else "bai_tap")


def load(name):
    p = SRC / f"{name}.py"
    if not p.exists():
        pytest.skip(f"chưa có {p}")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError:
        pytest.skip(f"{name}: chưa cài đặt")
    return mod



class TestSegMatchLosses:
    def setup_method(self):
        self.l = load("06_seg_match_losses")
        torch.manual_seed(0)
        self.lg, self.y = torch.randn(64, 3), torch.randint(0, 3, (64,))
        self.sl = torch.randn(4, 1, 16, 16)
        self.tg = (torch.rand(4, 1, 16, 16) > 0.8).float()

    def test_dice_zero_on_perfect_prediction(self):
        perf = self.tg * 20.0 - (1 - self.tg) * 20.0
        assert self.l.dice_loss(perf, self.tg).item() < 1e-4
        assert self.l.dice_loss(-perf, self.tg).item() > 0.9, "đoán ngược -> loss gần 1"

    def test_tversky_reduces_to_dice(self):
        """alpha=beta=0.5 quy về Dice — kiểm với eps NHỎ.
        Với eps=1.0 hai công thức lệch ~4e-3 vì eps vào mẫu theo hệ số khác nhau."""
        d = self.l.dice_loss(self.sl, self.tg, eps=1e-6)
        t = self.l.tversky_loss(self.sl, self.tg, 0.5, 0.5, eps=1e-6)
        assert abs(d.item() - t.item()) < 1e-5

    def test_tversky_beta_favours_recall(self):
        """beta > alpha phạt FN nặng hơn -> loss cao hơn khi model BỎ SÓT."""
        under = self.tg * 20.0 - 20.0            # dự đoán gần như toàn 0 -> nhiều FN
        hi_fn = self.l.tversky_loss(under, self.tg, alpha=0.1, beta=0.9)
        hi_fp = self.l.tversky_loss(under, self.tg, alpha=0.9, beta=0.1)
        assert hi_fn > hi_fp, "beta cao phải phạt bỏ sót nặng hơn"

    def test_contrastive_pulls_and_pushes(self):
        a = torch.zeros(4, 8); b = torch.zeros(4, 8)
        same = self.l.contrastive_loss(a, b, torch.ones(4), margin=1.0)
        assert same.item() < 1e-6, "cùng cặp + khoảng cách 0 -> loss 0"
        diff = self.l.contrastive_loss(a, b, torch.zeros(4), margin=1.0)
        # dung sai 1e-4 chu khong phai 1e-6: F.pairwise_distance cong eps=1e-6 noi bo
        # nen khoang cach khong bao gio bang 0 tuyet doi.
        assert abs(diff.item() - 1.0) < 1e-4, "khác cặp + khoảng cách 0 -> loss = margin^2"

    def test_triplet_zero_when_well_separated(self):
        a = torch.zeros(4, 8); p = torch.zeros(4, 8)
        n = torch.full((4, 8), 10.0)
        assert self.l.triplet_loss(a, p, n, margin=1.0).item() < 1e-6


# ═══════════════════════════════ 06 — CONSISTENCY & ĐA NHIỆM  → dùng ở TUẦN 6
