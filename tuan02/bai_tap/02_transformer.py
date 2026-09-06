"""BT 02 — TRANSFORMER SEQ2SEQ PRE-LN.  ⏱ ~2,5h · Ngày N3–N4   🔴 BÀI NẶNG NHẤT TUẦN

Không dùng nn.Transformer / nn.TransformerEncoderLayer.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Transformer
"""
import math, sys
from pathlib import Path
import torch
import torch.nn as nn

# nạp MultiHeadAttention từ BT 01 của chính bạn
import importlib.util as _u
_s = _u.spec_from_file_location("_att", Path(__file__).parent / "01_attention.py")
_att = _u.module_from_spec(_s); _s.loader.exec_module(_att)
MultiHeadAttention, causal_mask = _att.MultiHeadAttention, _att.causal_mask


class PositionalEncoding(nn.Module):
    """Positional encoding hình sin, KHÔNG học.

        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Tính div bằng exp(arange(0,d,2) * (-log(10000)/d)) — ổn định số học hơn là luỹ thừa.

    ⚠️ Phải dùng self.register_buffer("pe", ...) chứ KHÔNG phải nn.Parameter:
       nó không học, nhưng cần đi theo .to(device) và nằm trong state_dict.
       Test kiểm tra bạn có 0 parameters và có buffer tên 'pe'.
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (B, L, d_model). Cộng pe[:, :L] rồi dropout."""
        raise NotImplementedError


class FeedForward(nn.Module):
    """Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class EncoderLayer(nn.Module):
    """🔴 PRE-LN:   x = x + Dropout(Sublayer(LayerNorm(x)))

    KHÁC với bài báo gốc (Post-LN: x = LayerNorm(x + Sublayer(x))).
    Vì sao chọn Pre-LN cho kỳ thi: huấn luyện ỔN ĐỊNH, không cần warmup dài,
    ít phân kỳ. Post-LN mạnh hơn một chút khi tune kỹ, nhưng bạn không có
    thời gian tune trong 6 tiếng.

    Hai sublayer: self-attention, rồi feed-forward. Mỗi cái một LayerNorm riêng.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, src_mask=None):
        raise NotImplementedError


class DecoderLayer(nn.Module):
    """BA sublayer, theo thứ tự:
        1. self-attention có mask NHÂN QUẢ   (nhìn chính chuỗi đích, chỉ quá khứ)
        2. cross-attention                    (query = decoder, key/value = memory encoder)
        3. feed-forward
    Vẫn Pre-LN, ba LayerNorm riêng.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        raise NotImplementedError


class Seq2SeqTransformer(nn.Module):
    """Model hoàn chỉnh.

    __init__(src_vocab, tgt_vocab, d_model=256, n_heads=4, d_ff=512,
             n_enc=3, n_dec=3, dropout=0.1, max_len=512, pad_id=0,
             tie_decoder_output=True, tie_src_tgt=False)

    Thành phần: src_emb, tgt_emb (padding_idx=pad_id), pos, ModuleList enc/dec,
    ln_enc, ln_dec (Pre-LN cần LayerNorm CUỐI ở mỗi stack), out = Linear(d_model, tgt_vocab, bias=False).

    🔴 TIED EMBEDDINGS:  self.out.weight = self.tgt_emb.weight
       Giảm ~1/3 tham số và thường TỐT HƠN khi dữ liệu ít (hai ma trận học cùng
       một không gian ngữ nghĩa). tie_src_tgt chỉ hợp lệ khi vocab CHUNG.

    Ba phương thức phải có:
      pad_mask(seq) -> (B,1,1,L) bool, True ở vị trí KHÁC pad
      encode(src)   -> (memory, src_mask)
      decode(tgt_in, memory, src_mask) -> logits (B, L, tgt_vocab)
      forward(src, tgt_in) = decode(tgt_in, *encode(src))

    ⚠️ HAI BẪY:
      1. Nhân embedding với sqrt(d_model) trước khi cộng positional encoding.
         Thiếu bước này thì tín hiệu vị trí lấn át tín hiệu từ.
      2. Mask của decoder phải KẾT HỢP nhân quả VÀ padding:
             cm = causal_mask(L)[None, None]      # (1,1,L,L)
             tm = cm & self.pad_mask(tgt_in)      # (B,1,L,L) nhờ broadcast
         Chỉ dùng causal mà quên pad -> model học chú ý vào ô trống.
         Chỉ dùng pad mà quên causal -> RÒ RỈ TƯƠNG LAI: train loss đẹp,
         dịch ra rác. Test `test_model_is_causal` bắt đúng lỗi này.
    """
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=4, d_ff=512,
                 n_enc=3, n_dec=3, dropout=0.1, max_len=512, pad_id=0,
                 tie_decoder_output=True, tie_src_tgt=False):
        super().__init__()
        raise NotImplementedError

    def pad_mask(self, seq):
        raise NotImplementedError

    def encode(self, src):
        raise NotImplementedError

    def decode(self, tgt_in, memory, src_mask):
        raise NotImplementedError

    def forward(self, src, tgt_in):
        raise NotImplementedError
