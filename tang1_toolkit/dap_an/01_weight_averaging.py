"""Đáp án — Họ "trung bình trọng số": EMA · SWA · Checkpoint averaging."""
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
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def apply_to(self, model):
        """Đưa trọng số EMA vào model, giữ bản gốc để khôi phục."""
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(model.state_dict()[k].dtype)
                                  for k, v in self.shadow.items()}}, strict=False)

    def restore(self, model):
        if self.backup:
            model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
            self.backup = {}


def average_state_dicts(state_dicts):
    """Checkpoint averaging: trung bình CỘNG các state_dict (Ott 2018).

    Chỉ trung bình tensor số thực; tensor nguyên (num_batches_tracked của BatchNorm)
    lấy từ bản cuối. Đây là mẹo gần như luôn dương với chi phí bằng 0 — chỉ tốn
    dung lượng lưu vài checkpoint cuối.
    """
    assert len(state_dicts) > 0
    out = OrderedDict()
    for k in state_dicts[0]:
        vs = [sd[k] for sd in state_dicts]
        if vs[0].dtype.is_floating_point:
            out[k] = torch.stack([v.float() for v in vs]).mean(0).to(vs[0].dtype)
        else:
            out[k] = vs[-1]
    return out


class SWA:
    """Stochastic Weight Averaging: trung bình CHẠY (không mũ) các trọng số
    thu thập ở cuối mỗi epoch, thường trong giai đoạn LR cao & phẳng.

        avg = avg + (w - avg) / n
    """
    def __init__(self, model):
        self.n = 0
        self.avg = {k: v.detach().clone().float()
                    for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        for k, v in model.state_dict().items():
            if k in self.avg:
                self.avg[k].add_((v.detach().float() - self.avg[k]) / self.n)

    def state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.avg.items():
            sd[k] = v.to(sd[k].dtype)
        return sd


@torch.no_grad()
def update_bn(model, loader, device="cpu"):
    """SWA/checkpoint-averaging làm running stats của BatchNorm SAI, vì trọng số
    trung bình chưa từng "thấy" dữ liệu. Phải quét lại một lượt để tính lại.
    Model không có BatchNorm thì hàm này là no-op."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device))
    return model
