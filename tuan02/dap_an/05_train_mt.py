"""Đáp án — BT 05: Huấn luyện NMT đầu-cuối trên corpus Kơtu → Việt."""
import math, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).parent
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _load(name):
    m = __import__("importlib.util", fromlist=["util"])
    s = m.spec_from_file_location(name, HERE / f"{name}.py")
    mod = m.module_from_spec(s); s.loader.exec_module(mod)
    return mod


def read_tsv(path):
    rows = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    return [tuple(r.split("\t")) for r in rows if "\t" in r]


def build_vocab(token_lists, max_size=None):
    from collections import Counter
    c = Counter(t for toks in token_lists for t in toks)
    items = [w for w, _ in c.most_common(max_size)]
    itos = SPECIALS + items
    return {w: i for i, w in enumerate(itos)}, itos


class MTDataset(Dataset):
    def __init__(self, pairs, src_enc, tgt_enc, src_vocab, tgt_vocab):
        self.data = []
        for s, t in pairs:
            si = [src_vocab.get(x, UNK) for x in src_enc(s)]
            ti = [tgt_vocab.get(x, UNK) for x in tgt_enc(t)]
            self.data.append((si, [BOS] + ti + [EOS]))

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate(batch):
    """Đệm PAD về cùng độ dài. tgt_in = bỏ token cuối, tgt_out = bỏ token đầu."""
    sl = max(len(s) for s, _ in batch)
    tl = max(len(t) for _, t in batch)
    src = torch.full((len(batch), sl), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), tl), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s); tgt[i, :len(t)] = torch.tensor(t)
    return src, tgt[:, :-1], tgt[:, 1:]


def noam_lambda(d_model, warmup):
    """Lịch học gốc của bài Transformer: tăng tuyến tính rồi giảm theo 1/sqrt(step)."""
    def fn(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
    return fn


def train_model(model, train_dl, dev_dl, *, epochs=20, lr=1.0, warmup=400,
                label_smoothing=0.1, clip=1.0, device="cpu", log_every=0):
    model.to(device)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lambda(model.d_model, warmup))
    best, best_state = math.inf, None
    for ep in range(1, epochs + 1):
        model.train(); tot = n = 0
        for src, tin, tout in train_dl:
            src, tin, tout = src.to(device), tin.to(device), tout.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(src, tin)
            loss = lossf(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); sched.step()
            tot += loss.item() * tout.ne(PAD).sum().item(); n += tout.ne(PAD).sum().item()
        model.eval(); dtot = dn = 0
        with torch.inference_mode():
            for src, tin, tout in dev_dl:
                src, tin, tout = src.to(device), tin.to(device), tout.to(device)
                l = lossf(model(src, tin).reshape(-1, model.out.out_features), tout.reshape(-1))
                dtot += l.item() * tout.ne(PAD).sum().item(); dn += tout.ne(PAD).sum().item()
        dev_loss = dtot / dn
        if log_every and ep % log_every == 0:
            print(f"ep{ep:03d} train {tot/n:.4f} dev {dev_loss:.4f} lr {sched.get_last_lr()[0]:.2e}")
        if dev_loss < best:
            best = dev_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, best
