"""Đáp án — BT 02: Transformer seq2seq Pre-LN, tied embeddings."""
import math, sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util as _u
_s = _u.spec_from_file_location("_att", Path(__file__).parent / "01_attention.py")
_att = _u.module_from_spec(_s); _s.loader.exec_module(_att)
MultiHeadAttention, causal_mask = _att.MultiHeadAttention, _att.causal_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal, KHÔNG học. Lưu vào buffer để .to(device) mang theo và
    state_dict giữ được — nhưng không phải tham số."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """PRE-LN: x + Sublayer(LN(x)). Huấn luyện ổn định, KHÔNG cần warmup dài."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, src_mask)[0])
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, tgt_mask)[0])          # causal
        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, src_mask)[0])  # nhìn encoder
        x = x + self.drop(self.ff(self.ln3(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=4, d_ff=512,
                 n_enc=3, n_dec=3, dropout=0.1, max_len=512, pad_id=0,
                 tie_decoder_output=True, tie_src_tgt=False):
        super().__init__()
        self.pad_id, self.d_model = pad_id, d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        if tie_src_tgt:
            assert src_vocab == tgt_vocab, "chia sẻ embedding cần vocab CHUNG"
            self.tgt_emb = self.src_emb
        else:
            self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.enc = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.ln_enc, self.ln_dec = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_decoder_output:
            self.out.weight = self.tgt_emb.weight        # buộc trọng số: giảm ~1/3 tham số

    def pad_mask(self, seq):
        return (seq != self.pad_id)[:, None, None, :]    # (B,1,1,L) True = được nhìn

    def encode(self, src):
        m = self.pad_mask(src)
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.enc:
            x = layer(x, m)
        return self.ln_enc(x), m

    def decode(self, tgt_in, memory, src_mask):
        L = tgt_in.size(1)
        cm = causal_mask(L, tgt_in.device)[None, None]        # (1,1,L,L)
        # (1,1,L,L) & (B,1,1,L) -> (B,1,L,L): vừa nhân quả, vừa không nhìn vào pad
        tm = cm & self.pad_mask(tgt_in)
        x = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        for layer in self.dec:
            x = layer(x, memory, tm, src_mask)
        return self.out(self.ln_dec(x))

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)
