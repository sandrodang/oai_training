"""Bộ chấm tự động Tuần 4 — phần Tầng 1 (bộ đồ nghề học sâu).

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



class TestFocalLoss:
    def setup_method(self):
        self.l = load("06_focal_loss")
        torch.manual_seed(0)
        self.lg, self.y = torch.randn(64, 3), torch.randint(0, 3, (64,))
        self.sl = torch.randn(4, 1, 16, 16)
        self.tg = (torch.rand(4, 1, 16, 16) > 0.8).float()

    def test_focal_gamma0_equals_cross_entropy(self):
        assert torch.allclose(self.l.focal_loss(self.lg, self.y, gamma=0.0),
                              F.cross_entropy(self.lg, self.y), atol=1e-6), \
            "gamma=0 PHẢI quy về cross-entropy — nếu không, công thức sai"

    def test_focal_downweights_easy_examples(self):
        assert self.l.focal_loss(self.lg, self.y, gamma=2.0).item() < \
               F.cross_entropy(self.lg, self.y).item()

