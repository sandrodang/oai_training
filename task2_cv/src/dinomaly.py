"""Tai tao dac trung co nut that (Dinomaly-style) tren encoder DINOv2 dong bang.

Y tuong: decoder nhin toan anh qua attention NHUNG bi ep qua nut that hep + dropout,
nen no chi tai tao tot nhung gi thuong gap trong du lieu normal. Vung bat thuong
tai tao kem -> cosine distance cao.
Tham so hoc tu du lieu = trong so decoder (luu lam checkpoint).
"""
import os, sys, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR, CKPT_DIR, set_seed, holdout_split, read_test_csv
from features import ImgDS, get_model
from torch.utils.data import DataLoader

class Block(nn.Module):
    def __init__(self, d, h=8, mlp=4.0, drop=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True, dropout=drop)
        self.mlp = nn.Sequential(nn.Linear(d, int(d*mlp)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(d*mlp), d), nn.Dropout(drop))
    def forward(self, x):
        y = self.n1(x); x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class Decoder(nn.Module):
    """nut that hep + vai block transformer -> chan hoc anh xa dong nhat."""
    def __init__(self, d_in, d=512, nblk=4, drop=0.2, ngroup=2):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(drop),
                                        nn.Linear(d, d), nn.Dropout(drop))
        self.blocks = nn.ModuleList([Block(d, drop=drop*0.5) for _ in range(nblk)])
        self.heads = nn.ModuleList([nn.Linear(d, d_in // ngroup) for _ in range(ngroup)])
    def forward(self, x):
        h = self.bottleneck(x)
        for b in self.blocks: h = b(h)
        return [hd(h) for hd in self.heads]

def cos_loss(pred, tgt, hard_q=0.9):
    """1 - cosine, chi backprop tren cac diem KHO nhat (hard mining kieu Dinomaly)."""
    d = 1 - F.cosine_similarity(pred, tgt, dim=-1)          # B,N
    if hard_q < 1.0:
        thr = torch.quantile(d.detach().flatten().float(), 1 - hard_q)
        d = d[d >= thr]
    return d.mean()

@torch.no_grad()
def encode(files, long_side, dev, bs, layers, model_name="dinov2l_ml"):
    """-> [n_img, n_patch, ngroup*D] fp16 tren CPU (2 nhom tang, moi nhom trung binh)."""
    m, meta = get_model(model_name, dev)
    dl = DataLoader(ImgDS(files, long_side, meta["mult"]), batch_size=bs,
                    shuffle=False, num_workers=8, pin_memory=True)
    out = []
    for x in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m.get_intermediate_layers(x, n=layers, norm=True)
            g1 = torch.stack(o[:len(o)//2]).mean(0)
            g2 = torch.stack(o[len(o)//2:]).mean(0)
        out.append(torch.cat([g1, g2], -1).half().cpu())
    return torch.cat(out)

def train_decoder(feat, dev, iters, bs, d_model, nblk, drop, lr, seed=1337):
    torch.manual_seed(seed)
    D = feat.shape[-1]
    dec = Decoder(D, d_model, nblk, drop).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    n = feat.shape[0]
    g = torch.Generator().manual_seed(seed)
    dec.train()
    for it in range(iters):
        idx = torch.randint(n, (bs,), generator=g)
        x = feat[idx].to(dev).float()
        t1, t2 = x.chunk(2, dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p1, p2 = dec(x)
            loss = 0.5 * (cos_loss(p1.float(), t1) + cos_loss(p2.float(), t2))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); sch.step()
    return dec

@torch.no_grad()
def score(dec, feat, dev, bs=8, topq=0.01, keep_top=512):
    dec.eval(); sc, tops = [], []
    for i in range(0, feat.shape[0], bs):
        x = feat[i:i+bs].to(dev).float()
        t1, t2 = x.chunk(2, dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p1, p2 = dec(x)
        d = (1 - F.cosine_similarity(p1.float(), t1, -1)) + (1 - F.cosine_similarity(p2.float(), t2, -1))
        N = d.shape[1]; k = max(1, int(round(N * topq)))
        sc.append(torch.topk(d, k, 1).values.mean(1).cpu())
        tops.append(torch.topk(d, min(keep_top, N), 1).values.cpu())
    return torch.cat(sc).numpy(), torch.cat(tops).numpy().astype("float32"), feat.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=int, default=518)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--enc-bs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--nblk", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--topq", type=float, default=0.01)
    ap.add_argument("--layers", default="6,8,10,12")
    ap.add_argument("--enc", default="dinov2l_ml")
    ap.add_argument("--cats", default="")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    set_seed()
    dev = torch.device("cuda")
    layers = [int(x) for x in a.layers.split(",")]
    ck = os.path.join(CKPT_DIR, a.tag); os.makedirs(ck, exist_ok=True)
    res = {"_config": np.array([repr(vars(a))])}
    for cat in (a.cats.split(",") if a.cats else CATS):
        t0 = time.time()
        fit_f, hold_f = holdout_split(cat)
        ff = encode(fit_f, a.long, dev, a.enc_bs, layers, a.enc)
        dec = train_decoder(ff, dev, a.iters, a.bs, a.d_model, a.nblk, a.drop, a.lr)
        torch.save(dec.state_dict(), os.path.join(ck, f"{cat}_dec.pt"))
        del ff; torch.cuda.empty_cache()
        hf = encode(hold_f, a.long, dev, a.enc_bs, layers, a.enc)
        res[f"{cat}/hold"], res[f"{cat}/hold_top"], npat = score(dec, hf, dev, topq=a.topq)
        res[f"{cat}/hold_m"] = res[f"{cat}/hold"]
        res[f"{cat}/npatch"] = np.array([npat])
        rows = [r for r in read_test_csv("public") if r[1] == cat]
        tf = encode([r[2] for r in rows], a.long, dev, a.enc_bs, layers, a.enc)
        res[f"{cat}/public_ids"] = np.array([r[0] for r in rows])
        res[f"{cat}/public"], res[f"{cat}/public_top"], _ = score(dec, tf, dev, topq=a.topq)
        res[f"{cat}/public_m"] = res[f"{cat}/public"]
        h, t = res[f"{cat}/hold"], res[f"{cat}/public"]
        print(f"  {cat}: {time.time()-t0:.0f}s | hold med={np.median(h):.4f} test med={np.median(t):.4f} "
              f"sep={(np.median(t)-np.median(h))/(h.std()+1e-9):+.2f}sd", flush=True)
        del hf, tf; torch.cuda.empty_cache()
    np.savez(os.path.join(CACHE_DIR, f"scores_{a.tag}.npz"), **res)
    print("saved", a.tag)

if __name__ == "__main__":
    main()
