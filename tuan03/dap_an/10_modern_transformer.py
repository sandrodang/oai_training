"""BT 10 — TRANSFORMER HIỆN ĐẠI: RMSNorm · RoPE · GQA · SwiGLU.  Tuần 3

Bốn thành phần thay cho bản gốc 2017. Mô hình tham chiếu của tutorial chính thức
vòng Bắc 2025 dùng cả bốn.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Chỉ chuẩn hoá theo RMS — KHÔNG trừ mean, KHÔNG có bias."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * (x / rms)


def build_rope_cache(seq_len, dim, base=10000.0, device=None):
    """Trả (cos, sin) shape (seq_len, dim//2) cho quy ước cặp (2i, 2i+1)."""
    if dim % 2:
        raise ValueError("dim phải chẵn")
    i = torch.arange(dim // 2, dtype=torch.float32, device=device)
    theta = base ** (-2.0 * i / dim)
    m = torch.arange(seq_len, dtype=torch.float32, device=device)
    ang = m[:, None] * theta[None, :]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (..., L, d). Xoay từng CẶP LIỀN KỀ (2i, 2i+1) theo góc m*theta_i."""
    L, d = x.shape[-2], x.shape[-1]
    cos, sin = cos[:L], sin[:L]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def repeat_kv(x, n_rep):
    """(B, H_kv, L, dh) -> (B, H_kv*n_rep, L, dh), lặp LIỀN KHỐI theo head."""
    if n_rep == 1:
        return x
    B, H, L, dh = x.shape
    return x[:, :, None].expand(B, H, n_rep, L, dh).reshape(B, H * n_rep, L, dh)


def grouped_query_attention(q, k, v, mask=None):
    """q: (B, Hq, L, dh) · k,v: (B, Hkv, S, dh) với Hq chia hết cho Hkv."""
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} phải chia hết cho Hkv={Hkv}")
    k, v = repeat_kv(k, Hq // Hkv), repeat_kv(v, Hq // Hkv)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(-1)
    return attn @ v, attn


class FFN_SwiGLU(nn.Module):
    """SwiGLU: (SiLU(W_gate x)) * (W_up x) -> W_down.  Ba ma trận, không bias."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
