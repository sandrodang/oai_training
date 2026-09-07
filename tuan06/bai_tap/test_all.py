"""Bộ chấm tự động Tuần 6 — phần Tầng 1 (bộ đồ nghề học sâu).

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


def tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))



class TestFinetuneLR:
    def setup_method(self):
        self.f = load("06_finetune_lr")

    def test_no_decay_group_holds_only_1d_params(self):
        gs = self.f.param_groups_no_decay(tiny(), weight_decay=0.01)
        assert len(gs) == 2
        nd = [g for g in gs if g["weight_decay"] == 0.0][0]
        assert all(p.ndim <= 1 for p in nd["params"])
        assert len(nd["params"]) == 4, "2 bias Linear + weight&bias BatchNorm"

    def test_llrd_increases_with_depth(self):
        layers = [nn.Linear(8, 8) for _ in range(4)]
        head = nn.Linear(8, 2)
        gs = self.f.llrd_param_groups(layers, head, base_lr=2e-5, decay=0.5)
        lrs = [g["lr"] for g in gs]
        assert min(lrs) < max(lrs), "tầng thấp phải có LR NHỎ HƠN tầng cao"
        assert abs(max(lrs) - 2e-5) < 1e-12, "head phải giữ base_lr"
        assert abs(min(lrs) - 2e-5 * 0.5 ** 3) < 1e-12, "tầng thấp nhất = base * decay^(L-1)"

    def test_gradual_unfreeze_is_monotonic(self):
        sch = self.f.gradual_unfreeze_schedule(6, 4)
        assert sch[0] == 0 and sch[-1] == 6, "bắt đầu chỉ head, cuối mở hết"
        assert all(a <= b for a, b in zip(sch, sch[1:])), "phải không giảm"

    def test_freeze_batchnorm(self):
        m = tiny()
        n = self.f.freeze_batchnorm(m)
        bn = m[1]
        assert n == 1 and not bn.training
        assert all(not p.requires_grad for p in bn.parameters())


# ═══════════════════════════════ 04 — ĐỐI KHÁNG  → dùng ở TUẦN 6
class _EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.emb(x).mean(1))
