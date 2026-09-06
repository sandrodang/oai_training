"""Multi-task transformer: shared encoder -> hate head (3) + noise head (7).

Runs on GPU. Fully seeded. `id` is never fed to the model.
"""
import sys, os, time, math, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
import random
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from common import *

p = argparse.ArgumentParser()
p.add_argument("--model", default="5CD-AI/visobert-14gb-corpus")
p.add_argument("--tag", default="visobert")
p.add_argument("--folds", default="0")        # "0" | "all"
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--head_lr", type=float, default=1e-3)
p.add_argument("--maxlen", type=int, default=128)
p.add_argument("--w_noise", type=float, default=0.3)
p.add_argument("--seed", type=int, default=SEED)
p.add_argument("--test", default="public_test")
p.add_argument("--view", default="raw", choices=["raw","dual"])
p.add_argument("--save_ckpt", type=int, default=1)
p.add_argument("--hier", type=int, default=0)     # hierarchical hate head
p.add_argument("--llrd", type=float, default=1.0) # layer-wise lr decay (1.0 = off)
p.add_argument("--ema", type=float, default=0.0)  # EMA decay (0 = off)
p.add_argument("--fgm", type=float, default=0.0)  # adversarial perturbation on the embeddings
p.add_argument("--loss", default="ce", choices=["ce", "gce", "ls"])
p.add_argument("--loss_n", default=None, choices=["ce", "gce", "ls"])  # noise head, defaults to --loss
p.add_argument("--gce_q", type=float, default=0.7)
p.add_argument("--ls", type=float, default=0.1)
p.add_argument("--aug", type=float, default=0.0)   # prob. of noising a training row
p.add_argument("--cons", type=float, default=0.0)  # consistency weight on the REAL (original, noised) pairs
# parse only when run directly -- importing this module (predict.py does)
# must not consume the importer's sys.argv
a = p.parse_args() if __name__ == "__main__" else p.parse_args([])

set_seed(a.seed)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


class DS(Dataset):
    def __init__(self, texts, tok, maxlen, yh=None, yn=None, pair=None, aug=None, augp=0.0):
        self.t, self.tok, self.m, self.yh, self.yn = list(texts), tok, maxlen, yh, yn
        self.pair = pair            # index of the same source comment, other noise form
        self.aug, self.augp = aug, augp
    def __len__(self): return len(self.t)
    def _enc(self, i, noised=False):
        x = self.t[i]
        if noised and not isinstance(x, tuple):
            x = self.aug(x)
        return (self.tok(x[0], x[1], truncation=True, max_length=self.m)
                if isinstance(x, tuple) else self.tok(x, truncation=True, max_length=self.m))
    def __getitem__(self, i):
        did = self.aug is not None and self.augp > 0 and random.random() < self.augp
        e = self._enc(i, did)
        d = {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
        if self.yh is not None:
            d["yh"] = int(self.yh[i]); d["yn"] = int(self.yn[i])
            d["nmask"] = 0.0 if did else 1.0      # hate label survives, noise label does not
        if self.pair is not None:
            j = self.pair[i]
            j = i if j < 0 else int(j)
            e2 = self._enc(j)
            d["p_input_ids"] = e2["input_ids"]; d["p_attention_mask"] = e2["attention_mask"]
            d["p_yh"] = int(self.yh[j]); d["p_yn"] = int(self.yn[j])
            d["p_ok"] = float(self.pair[i] >= 0)
        return d


def _pad(batch, key, pad_id):
    L = max(len(b[key]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b[key])] = torch.tensor(b[key])
    return ids


def collate(batch, pad_id):
    out = {"input_ids": _pad(batch, "input_ids", pad_id),
           "attention_mask": _pad(batch, "attention_mask", 0)}
    if "yh" in batch[0]:
        out["yh"] = torch.tensor([b["yh"] for b in batch])
        out["yn"] = torch.tensor([b["yn"] for b in batch])
        out["nmask"] = torch.tensor([b.get("nmask", 1.0) for b in batch])
    if "p_input_ids" in batch[0]:
        out["p_input_ids"] = _pad(batch, "p_input_ids", pad_id)
        out["p_attention_mask"] = _pad(batch, "p_attention_mask", 0)
        out["p_yh"] = torch.tensor([b["p_yh"] for b in batch])
        out["p_yn"] = torch.tensor([b["p_yn"] for b in batch])
        out["p_ok"] = torch.tensor([b["p_ok"] for b in batch])
    return out


def gce(logits, target, weight, q=0.7):
    """Generalised Cross Entropy -- bounded loss that stops the model chasing
    mislabelled examples. The organisers confirmed the noise_type labels record
    the operator rather than the surface form, so part of the data is provably
    mislabelled relative to what is observable; the hate labels carry the usual
    UIT-ViHSD OFFENSIVE/HATE annotation ambiguity on top."""
    p = logits.softmax(-1).gather(1, target[:, None]).squeeze(1).clamp(1e-6, 1.0)
    l = (1.0 - p.pow(q)) / q
    w = weight[target]
    return (l * w).sum() / w.sum()


def crit(logits, target, weight, a, mask=None, kind=None):
    kind = kind or a.loss
    if mask is not None:
        if mask.sum() < 1:
            return logits.sum() * 0.0
        keep = mask > 0.5
        logits, target = logits[keep], target[keep]
    if kind == "gce":
        return gce(logits, target, weight, a.gce_q)
    if kind == "ls":
        return F.cross_entropy(logits, target, weight=weight, label_smoothing=a.ls)
    return F.cross_entropy(logits, target, weight=weight)


class EMA:
    """Exponential moving average of the weights. FGM peaks around epoch 5 and
    then oscillates (0.7277 -> 0.7180 by epoch 8), i.e. the optimiser is
    circling a minimum -- exactly the case averaging is meant to smooth."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.bak = {}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    def swap_in(self, model):
        self.bak = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: p.data.copy_(self.shadow[n])
    def swap_out(self, model):
        for n, p in model.named_parameters():
            if n in self.bak: p.data.copy_(self.bak[n])
        self.bak = {}


class FGM:
    """Fast Gradient Method: perturb the word embeddings along their gradient
    and take a second backward pass. Directly targets robustness, which is
    exactly what this task is scored on."""
    def __init__(self, model, eps=1.0, key="word_embeddings"):
        self.m, self.eps, self.key, self.bak = model, eps, key, {}
    def attack(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad and self.key in n and p.grad is not None:
                self.bak[n] = p.data.clone()
                nrm = torch.norm(p.grad)
                if nrm != 0 and not torch.isnan(nrm):
                    p.data.add_(self.eps * p.grad / nrm)
    def restore(self):
        for n, p in self.m.named_parameters():
            if n in self.bak:
                p.data = self.bak[n]
        self.bak = {}


class MTL(nn.Module):
    def __init__(self, name, hier=0):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.hier  = hier
        # 3-way head, or: CLEAN-vs-TOXIC then OFFENSIVE-vs-HATE.
        # The measured error profile is toxic->CLEAN (29% of OFFENSIVE), not
        # OFFENSIVE<->HATE, so the binary boundary deserves its own classifier
        # trained on the full data rather than one third of a 3-way softmax.
        if hier:
            self.toxic = nn.Linear(h * 2, 1)
            self.split = nn.Linear(h * 2, 2)
        else:
            self.hate = nn.Linear(h * 2, 3)
        self.noise = nn.Linear(h * 2, 7)
    def forward(self, ids, att):
        o = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        mean = (o * m).sum(1) / m.sum(1).clamp(min=1e-6)
        mx = (o.masked_fill(m == 0, -1e4)).max(1).values
        z = self.drop(torch.cat([mean, mx], -1))
        if not self.hier:
            return self.hate(z), self.noise(z)
        # compose exact 3-way log-probabilities; log_softmax of a log-prob vector
        # is the identity, so cross_entropy downstream stays correct unchanged
        lt = F.logsigmoid(self.toxic(z)).squeeze(-1)          # log P(toxic)
        ln_ = F.logsigmoid(-self.toxic(z)).squeeze(-1)        # log P(clean)
        sp = self.split(z).log_softmax(-1)                    # log P(off|tox), P(hate|tox)
        return torch.stack([ln_, lt + sp[:, 0], lt + sp[:, 1]], -1), self.noise(z)


def run_fold(df, k, tok, te_texts):
    trn, val = df[df.fold != k], df[df.fold == k]
    pad = tok.pad_token_id
    cf = lambda b: collate(b, pad)
    pair = None
    if a.cons > 0:
        pos = {ix: k for k, ix in enumerate(trn.index)}
        pair = np.full(len(trn), -1, dtype=np.int64)
        for _, sub in trn.groupby("group"):
            if len(sub) < 2:
                continue
            ori = sub.index[sub.noise_type.values == "ORIGINAL"]
            for ix in sub.index:
                cand = [c for c in (ori if len(ori) else sub.index) if c != ix]
                if cand:
                    pair[pos[ix]] = pos[cand[0]]
        print(f"  fold{k} consistency pairs: {(pair >= 0).sum()}/{len(trn)}", flush=True)
    augr = None
    if a.aug > 0:
        from augment import Augmenter, build as build_aug
        augr = Augmenter(build_aug(), random.Random(a.seed + k))
        print(f"  fold{k} augmentation p={a.aug}", flush=True)
    dl_tr = DataLoader(DS(list(trn["view"]), tok, a.maxlen, trn.y_hate.values,
                          trn.y_noise.values, pair, augr, a.aug),
                       batch_size=a.bs, shuffle=True, collate_fn=cf, num_workers=4, drop_last=True)
    dl_va = DataLoader(DS(list(val["view"]), tok, a.maxlen), batch_size=256, collate_fn=cf, num_workers=4)
    dl_te = DataLoader(DS(te_texts, tok, a.maxlen), batch_size=256, collate_fn=cf, num_workers=4)

    model = MTL(a.model, a.hier).to(dev)
    wh = torch.tensor(len(trn) / (3 * np.bincount(trn.y_hate, minlength=3)), dtype=torch.float, device=dev)
    wn = torch.tensor(len(trn) / (7 * np.bincount(trn.y_noise, minlength=7)), dtype=torch.float, device=dev)
    HEADS = ("hate", "noise", "toxic", "split")
    head = [n for n, _ in model.named_parameters() if n.startswith(HEADS)]
    if a.llrd < 1.0:
        nl = model.enc.config.num_hidden_layers
        groups = [{"params": [q for n, q in model.named_parameters() if n in head], "lr": a.head_lr}]
        for li in range(nl, -1, -1):
            key = f"encoder.layer.{li}." if li < nl else "embeddings."
            ps = [q for n, q in model.named_parameters() if key in n and n not in head]
            if ps:
                groups.append({"params": ps, "lr": a.lr * (a.llrd ** (nl - li))})
        used = {id(q) for g in groups for q in g["params"]}
        rest = [q for n, q in model.named_parameters() if id(q) not in used]
        if rest:
            groups.append({"params": rest, "lr": a.lr})
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW([
            {"params": [q for n, q in model.named_parameters() if n not in head], "lr": a.lr},
            {"params": [q for n, q in model.named_parameters() if n in head], "lr": a.head_lr},
        ], weight_decay=0.01)
    steps = len(dl_tr) * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    fgm = FGM(model, a.fgm) if a.fgm > 0 else None
    ema = EMA(model, a.ema) if a.ema > 0 else None

    def infer(dl):
        model.eval(); H, N = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for b in dl:
                lh, ln = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                H.append(lh.float().softmax(-1).cpu()); N.append(ln.float().softmax(-1).cpu())
        return torch.cat(H).numpy(), torch.cat(N).numpy()

    best = (-1, None, None)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for b in dl_tr:
            ids, att = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            th, tn = b["yh"].to(dev), b["yn"].to(dev)

            def _loss():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lh, ln = model(ids, att)
                    nm = b["nmask"].to(dev) if "nmask" in b else None
                    L = (1 - a.w_noise) * crit(lh, th, wh, a) \
                        + a.w_noise * crit(ln, tn, wn, a, nm, a.loss_n)
                    if a.cons > 0:
                        p_ids = b["p_input_ids"].to(dev); p_att = b["p_attention_mask"].to(dev)
                        ok = b["p_ok"].to(dev)
                        lh2, ln2 = model(p_ids, p_att)
                        L = 0.5 * L + 0.5 * ((1 - a.w_noise) * crit(lh2, b["p_yh"].to(dev), wh, a)
                                             + a.w_noise * crit(ln2, b["p_yn"].to(dev), wn, a))
                        # the two views are the SAME source comment, so their hate
                        # distribution must agree; the noise head must NOT be tied.
                        q1 = lh.float().log_softmax(-1); q2 = lh2.float().log_softmax(-1)
                        kl = 0.5 * (F.kl_div(q1, q2, log_target=True, reduction="none").sum(-1)
                                    + F.kl_div(q2, q1, log_target=True, reduction="none").sum(-1))
                        L = L + a.cons * (kl * ok).sum() / ok.sum().clamp(min=1)
                    return L

            opt.zero_grad(set_to_none=True)
            loss = _loss(); loss.backward()
            if fgm is not None:                      # second, adversarial pass
                fgm.attack(); _loss().backward(); fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
            if ema is not None: ema.update(model)
        if ema is not None: ema.swap_in(model)
        ph, pn = infer(dl_va)
        s, fh, fn = official_score(val.y_hate.values, ph.argmax(1), val.y_noise.values, pn.argmax(1))
        print(f"  fold{k} ep{ep+1} loss={tot/len(dl_tr):.4f} hate={fh:.4f} noise={fn:.4f} SCORE={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if s > best[0]:
            best = (s, (ph, pn), infer(dl_te))
            if a.save_ckpt:
                os.makedirs(f"ckpt/{a.tag}", exist_ok=True)
                torch.save({"model": model.state_dict(), "args": vars(a), "fold": k,
                            "epoch": ep + 1, "score": s}, f"ckpt/{a.tag}/fold{k}.pt")
        # restore the live weights: evaluation and the checkpoint use the EMA
        # copy, but training must continue from the real ones
        if ema is not None:
            ema.swap_out(model)
    return best, val.index.values


if __name__ == "__main__":
    df = load_labeled(); df["fold"] = make_folds(df)
    te = load_test(a.test)
    if a.view == "dual":
        from normalize import build as build_norm
        nz, st = build_norm(df)
        print(f"[norm] {st}", flush=True)
        df["view"] = [(t, nz(t)) for t in df.text]
        te_view = [(t, nz(t)) for t in te.text]
    else:
        df["view"] = df.text
        te_view = list(te.text)
    tok = AutoTokenizer.from_pretrained(a.model)
    ln = [len(tok(*t)["input_ids"]) if isinstance(t, tuple) else len(tok(t)["input_ids"]) for t in df["view"].sample(3000, random_state=0)]
    print(f"[tok] token len p50={np.percentile(ln,50):.0f} p95={np.percentile(ln,95):.0f} p99={np.percentile(ln,99):.0f} max_len={a.maxlen}", flush=True)

    ks = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    oof_h = np.zeros((len(df), 3)); oof_n = np.zeros((len(df), 7)); seen = np.zeros(len(df), bool)
    te_h, te_n = [], []
    for k in ks:
        (s, (vh, vn), (th, tn)), idx = run_fold(df, k, tok, te_view)
        oof_h[idx], oof_n[idx], seen[idx] = vh, vn, True
        te_h.append(th); te_n.append(tn)
        print(f"[fold {k}] best SCORE={s:.4f}", flush=True)

    m = seen
    s, fh, fn = official_score(df.y_hate.values[m], oof_h[m].argmax(1), df.y_noise.values[m], oof_n[m].argmax(1))
    print(f"\n[{a.tag}] OOF over {m.sum()} rows: hate={fh:.4f} noise={fn:.4f} SCORE={s:.4f}")
    np.savez(f"oof/{a.tag}_oof.npz", h=oof_h, n=oof_n, seen=seen,
             te_h=np.mean(te_h, 0), te_n=np.mean(te_n, 0))
    print(f"[saved] oof/{a.tag}_oof.npz")
