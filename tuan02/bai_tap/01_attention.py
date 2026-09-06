"""BT 01 — ATTENTION TỪ SỐ 0.  ⏱ ~1,5h · Ngày N2

Cấm dùng nn.MultiheadAttention hay F.scaled_dot_product_attention trong lời giải
(test SẼ dùng chúng làm chuẩn đối chiếu).
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Attention
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size, device=None):
    """Trả tensor bool (size, size). Quy ước: **True = ĐƯỢC nhìn**.

    Vị trí i chỉ được nhìn các vị trí j <= i  ->  tam giác dưới kể cả đường chéo.
    Gợi ý: torch.ones(...).tril()
    """
    raise NotImplementedError


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """q, k, v: (..., L, d_k).  mask broadcast được về (..., Lq, Lk), True = được nhìn.
    Trả (output, attn_weights).

        scores = q @ k^T / sqrt(d_k)
        scores[~mask] = -inf
        attn   = softmax(scores, dim=-1)
        out    = attn @ v

    ⚠️ VÌ SAO CHIA sqrt(d_k): q·k là tổng của d_k tích, phương sai tỉ lệ với d_k.
       Không chia thì với d_k=64 các score bị kéo ra rất xa nhau, softmax bão hoà,
       gradient gần như bằng 0. Đây là bug ÂM THẦM: model vẫn chạy, chỉ học rất kém.
       Test `test_scaling_by_sqrt_dk` bắt lỗi này bằng cách đo entropy.

    ⚠️ Dùng masked_fill với -inf TRƯỚC softmax, không phải nhân 0 SAU softmax.
    """
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """4 phép chiếu tuyến tính: W_q, W_k, W_v (d_model->d_model) và W_o (d_model->d_model).

    Luồng:
        1. chiếu q, k, v
        2. tách thành n_heads:  (B, L, d_model) -> (B, n_heads, L, d_head)
           dùng .view(B, L, H, dh).transpose(1, 2)
        3. gọi scaled_dot_product_attention
        4. gộp lại: .transpose(1,2).contiguous().view(B, L, d_model)
        5. qua W_o

    ⚠️ Nhớ .contiguous() trước .view() sau khi transpose, nếu không sẽ lỗi runtime.
    ⚠️ mask có thể vào dưới dạng (Lq,Lk) hoặc (B,1,Lq,Lk) hoặc (B,1,1,Lk).
       Nếu mask.dim()==2 thì thêm 2 chiều đầu: mask[None, None].
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, query, key, value, mask=None):
        """Trả (output (B,L,d_model), attn_weights (B,n_heads,Lq,Lk))."""
        raise NotImplementedError
