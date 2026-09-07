"""BT 00 — TORCH CƠ BẢN.  Tuần 1 · N5–N6

Mọi bài sau đều đứng trên file này. Bảy hàm dưới đây KHÔNG phải bài tập cú pháp —
mỗi hàm là một chỗ mà bản sai **vẫn chạy và vẫn ra số**.
"""
import torch
import torch.nn as nn


def batched_dot(a, b):
    """a, b: (B, D) -> (B,) tích vô hướng TỪNG HÀNG."""
    return (a * b).sum(dim=-1)


def add_feature_bias(x, bias):
    """x: (B, L, D) · bias: (D,) -> cộng bias theo chiều ĐẶC TRƯNG (cuối)."""
    if bias.dim() != 1 or bias.shape[0] != x.shape[-1]:
        raise ValueError(f"bias phải có shape ({x.shape[-1]},), đang {tuple(bias.shape)}")
    return x + bias


def split_heads(x, n_heads):
    """(B, L, D) -> (B, H, L, D//H).  view TRƯỚC, transpose SAU."""
    B, L, D = x.shape
    if D % n_heads:
        raise ValueError(f"D={D} không chia hết cho n_heads={n_heads}")
    return x.view(B, L, n_heads, D // n_heads).transpose(1, 2)


def merge_heads(x):
    """(B, H, L, dh) -> (B, L, H*dh). Nghịch đảo chính xác của split_heads."""
    B, H, L, dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * dh)


def masked_mean(x, mask):
    """x: (B, L, D) · mask: (B, L) bool, True = token THẬT.
    Trung bình theo chiều L, BỎ QUA ô đệm. Câu toàn pad -> vector 0.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    s = (x * m).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1e-9)
    return s / n


def grad_wrt(f, x):
    """Gradient của f(x) (vô hướng) theo x. Trả tensor cùng shape với x."""
    x = x.detach().clone().requires_grad_(True)
    f(x).backward()
    return x.grad


class RunningMean(nn.Module):
    """Trung bình luỹ kế — trạng thái phải là BUFFER, không phải Parameter.

    Buffer: nằm trong state_dict, đi theo .to(device), KHÔNG nằm trong .parameters()
    nên optimizer không đụng tới. Dùng Parameter ở đây thì optimizer sẽ "học"
    một con số vốn chỉ là thống kê -> hỏng âm thầm.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        self.total += x.sum()
        self.count += x.numel()
        return self.total / self.count.clamp(min=1)


def running_loss(losses):
    """Cộng dồn loss để LOG. Phải .detach() — nếu không, đồ thị tính toán của mọi
    batch bị giữ lại và bộ nhớ phình dần cho tới khi hết VRAM.
    Trả một float Python.
    """
    tot = 0.0
    for l in losses:
        tot += l.detach().item()
    return tot
