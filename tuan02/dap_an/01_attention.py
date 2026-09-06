"""Đáp án — BT 01: Attention từ số 0."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size, device=None):
    """(size, size) bool. True = ĐƯỢC nhìn. Tam giác dưới kể cả đường chéo."""
    return torch.ones(size, size, dtype=torch.bool, device=device).tril()


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """q,k,v: (..., L, d). mask: bool broadcast được về (..., Lq, Lk), True = được nhìn.
    Trả (output, attn_weights)."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)      # chia sqrt(d_k): giữ phương sai ~1
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)   # (B,H,L,dh)

    def _merge(self, x):
        B, H, L, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * dh)

    def forward(self, query, key, value, mask=None):
        """mask: (B,1,Lq,Lk) hoặc (Lq,Lk) hoặc (B,1,1,Lk). True = được nhìn."""
        q, k, v = self._split(self.w_q(query)), self._split(self.w_k(key)), self._split(self.w_v(value))
        if mask is not None and mask.dim() == 2:
            mask = mask[None, None, :, :]
        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout, self.training)
        return self.w_o(self._merge(out)), attn
