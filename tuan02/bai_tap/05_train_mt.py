"""BT 05 — HUẤN LUYỆN NMT ĐẦU-CUỐI.  ⏱ ~2h · Ngày N7

Ráp BT 01–04 lại và dịch Kơtu -> Việt. Ánh xạ là XÁC ĐỊNH, nên một pipeline đúng
PHẢI đạt BLEU > 40 (đáp án đạt ~80). BLEU thấp => có lỗi, không phải "bài khó".

Chấm nhanh:  python3 -m pytest bai_tap/test_all.py -q -k TrainMT -m "not slow"
Chấm đủ:     python3 -m pytest bai_tap/test_all.py -q -m slow -s     (~40s CPU)
"""
import math, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).parent
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def read_tsv(path):
    """Đọc file tsv có header 'src\\ttgt', trả list[(src, tgt)]."""
    raise NotImplementedError


def build_vocab(token_lists, max_size=None):
    """Trả (stoi, itos). 4 token đặc biệt đứng ĐẦU theo đúng thứ tự SPECIALS,
    sau đó là các token thường sắp theo tần suất giảm dần (Counter.most_common)."""
    raise NotImplementedError


class MTDataset(Dataset):
    """__getitem__ -> (src_ids, tgt_ids) với tgt_ids = [BOS] + ids + [EOS].
    Token lạ -> UNK. src KHÔNG cần bos/eos."""
    def __init__(self, pairs, src_enc, tgt_enc, src_vocab, tgt_vocab):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def collate(batch):
    """Đệm PAD, trả (src, tgt_in, tgt_out).

    🔴 DỊCH MỘT VỊ TRÍ — điểm dễ sai nhất của seq2seq:
        tgt      = [BOS, a, b, EOS]
        tgt_in   = [BOS, a, b]        <- ĐẦU VÀO decoder (bỏ token CUỐI)
        tgt_out  = [a, b, EOS]        <- NHÃN            (bỏ token ĐẦU)
    Tại vị trí i, decoder nhìn tgt_in[:i+1] và phải đoán tgt_out[i].
    Lệch một ô là model học "sao chép đầu vào" — loss giảm rất đẹp, BLEU bằng 0.
    """
    raise NotImplementedError


def noam_lambda(d_model, warmup):
    """Lịch học gốc của bài Transformer (Vaswani 2017, mục 5.3):

        lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Tăng TUYẾN TÍNH trong warmup bước, rồi giảm theo 1/sqrt(step).
    Dùng với torch.optim.lr_scheduler.LambdaLR và Adam(lr=1.0).
    Đỉnh rơi đúng tại step = warmup.
    """
    raise NotImplementedError


def train_model(model, train_dl, dev_dl, *, epochs=20, lr=1.0, warmup=400,
                label_smoothing=0.1, clip=1.0, device="cpu", log_every=0):
    """Trả (model đã nạp trạng thái tốt nhất theo dev loss, dev_loss tốt nhất).

    Phải có đủ:
      □ CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        - ignore_index=PAD: KHÔNG tính loss trên ô đệm, nếu không loss bị pha loãng
        - label_smoothing=0.1: chuẩn de facto cho MT, gần như luôn dương
      □ Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9)  <- lr=1.0 vì Noam tự chia tỉ lệ
      □ LambdaLR với noam_lambda(model.d_model, warmup)
      □ clip_grad_norm_ (seq2seq rất dễ nổ gradient ở những epoch đầu)
      □ logits.reshape(-1, V) và tgt_out.reshape(-1) trước khi tính loss
      □ theo dõi dev loss, giữ bản state_dict tốt nhất, cuối cùng nạp lại

    Mẹo: chuẩn hoá loss theo SỐ TOKEN KHÁC PAD, không phải số câu — nếu không
    các batch có câu dài sẽ bị tính nặng hơn một cách vô lý.
    """
    raise NotImplementedError
