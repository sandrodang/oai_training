"""BÀI TẬP — Họ "trung bình trọng số": EMA · SWA · Checkpoint averaging."""
import copy
from collections import OrderedDict
import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average của trọng số.

        shadow = decay * shadow + (1 - decay) * param

    Cập nhật SAU mỗi optimizer.step(). Lúc đánh giá thì swap shadow vào model.
    decay 0.999 ~ trung bình trượt trên ~1000 bước gần nhất.
    """
    def __init__(self, model, decay=0.999):
        raise NotImplementedError

    @torch.no_grad()
    def update(self, model):
        raise NotImplementedError

    def apply_to(self, model):
        """Đưa trọng số EMA vào model, giữ bản gốc để khôi phục."""
        raise NotImplementedError

    def restore(self, model):
        raise NotImplementedError


def average_state_dicts(state_dicts):
    """Checkpoint averaging: trung bình CỘNG các state_dict (Ott 2018).

    Chỉ trung bình tensor số thực; tensor nguyên (num_batches_tracked của BatchNorm)
    lấy từ bản cuối. Đây là mẹo gần như luôn dương với chi phí bằng 0 — chỉ tốn
    dung lượng lưu vài checkpoint cuối.
    """
    raise NotImplementedError


class SWA:
    """Stochastic Weight Averaging: trung bình CHẠY (không mũ) các trọng số
    thu thập ở cuối mỗi epoch, thường trong giai đoạn LR cao & phẳng.

        avg = avg + (w - avg) / n
    """
    def __init__(self, model):
        raise NotImplementedError

    @torch.no_grad()
    def update(self, model):
        raise NotImplementedError

    def state_dict(self, model):
        raise NotImplementedError


@torch.no_grad()
def update_bn(model, loader, device="cpu"):
    """SWA/checkpoint-averaging làm running stats của BatchNorm SAI, vì trọng số
    trung bình chưa từng "thấy" dữ liệu. Phải quét lại một lượt để tính lại.
    Model không có BatchNorm thì hàm này là no-op."""
    raise NotImplementedError
