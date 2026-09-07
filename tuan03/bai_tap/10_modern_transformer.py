"""BT 10 — TRANSFORMER HIỆN ĐẠI: RMSNorm · RoPE · GQA · SwiGLU.  ⏱ ~2h · Tuần 3

Tuần 2 bạn dựng Transformer bản gốc 2017 để HIỂU. Tuần này nâng cấp bốn thành phần
để SO — mô hình tham chiếu của tutorial chính thức vòng Bắc 2025 dùng cả bốn.

🔴 Vì sao đáng làm: nếu tác vụ NLP **cấm pretrained** (§11 câu 0 của kế hoạch — luật
vòng Bắc 2025 đúng như vậy), thì **chất lượng kiến trúc tự viết chính là chỗ cạnh tranh**.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k ModernTransformer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Chuẩn hoá chỉ theo RMS:  y = w * x / sqrt(mean(x²) + eps)

    ⚠️ KHÁC LayerNorm ở hai điểm, và cả hai đều hỏng ÂM THẦM nếu làm sai:
       - **KHÔNG trừ mean.** Trừ mean thì nó thành LayerNorm — vẫn chạy, vẫn hội tụ,
         chỉ là bạn không cài cái mình tưởng.
       - **KHÔNG có bias β.** Chỉ một tham số học được là `w` (khởi tạo 1).
    """
    def __init__(self, d, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_rope_cache(seq_len, dim, base=10000.0, device=None):
    """Trả (cos, sin), mỗi cái shape (seq_len, dim//2).

        theta_i = base^(-2i/dim)   với i = 0 .. dim/2-1
        ang[m, i] = m * theta_i
        cos = cos(ang), sin = sin(ang)

    Ném ValueError nếu dim lẻ.
    """
    raise NotImplementedError


def apply_rope(x, cos, sin):
    """x: (..., L, d). Xoay từng CẶP LIỀN KỀ (2i, 2i+1) theo góc m*theta_i:

        out[2i]   = x[2i]*cos - x[2i+1]*sin
        out[2i+1] = x[2i]*sin + x[2i+1]*cos

    🔴 VÌ SAO ĐÂY LÀ CẢ ĐIỂM CỦA RoPE: sau khi xoay, tích vô hướng q̃_m · k̃_n
       **chỉ phụ thuộc khoảng cách (m − n)**, không phụ thuộc vị trí tuyệt đối.
       Vị trí TƯƠNG ĐỐI được nhúng thẳng vào attention, không cần sửa softmax.

    ⚠️ BẪY ÂM THẦM: có HAI quy ước ghép cặp phổ biến —
       cặp liền kề `(2i, 2i+1)` (bài này, đúng công thức tutorial) và
       nửa-tách `(i, i+d/2)` (LLaMA). Dùng nhầm quy ước thì model **vẫn train được**,
       vẫn ra BLEU, chỉ là tín hiệu vị trí bị bóp méo. Test kiểm bằng tính chất
       tương đối ở trên, không kiểm từng phần tử — vì tính chất mới là thứ quan trọng.
    """
    raise NotImplementedError


def repeat_kv(x, n_rep):
    """(B, H_kv, L, dh) -> (B, H_kv*n_rep, L, dh).

    ⚠️ Phải lặp LIỀN KHỐI theo head: head kv thứ 0 phục vụ n_rep query head ĐẦU TIÊN.
       Dùng `x.repeat(1, n_rep, 1, 1)` là SAI — nó xếp xen kẽ, không báo lỗi.
       Cách đúng: `x[:, :, None].expand(B, H, n_rep, L, dh).reshape(B, H*n_rep, L, dh)`
    """
    raise NotImplementedError


def grouped_query_attention(q, k, v, mask=None):
    """q: (B, Hq, L, dh) · k, v: (B, Hkv, S, dh), Hq chia hết cho Hkv.

    Nhiều query head DÙNG CHUNG một cặp K/V head -> giảm bộ nhớ và chi phí giải mã.
    Hkv = Hq  -> đúng bằng multi-head attention thường.
    Hkv = 1   -> multi-query attention.

    Trả (output, attn). Ném ValueError nếu Hq không chia hết Hkv.
    Nhớ chia sqrt(dh) và masked_fill(-inf) TRƯỚC softmax (bài học Tuần 2).
    """
    raise NotImplementedError


class FFN_SwiGLU(nn.Module):
    """Thay Linear->ReLU->Linear bằng:   down( SiLU(gate(x)) * up(x) )

    BA ma trận, đều `bias=False`: gate (d_model->d_ff), up (d_model->d_ff),
    down (d_ff->d_model).

    ⚠️ Đảo nhánh — `gate(x) * SiLU(up(x))` — vẫn chạy, vẫn học, chỉ khác hàm.
       Nhánh đi qua SiLU là nhánh **gate**.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
