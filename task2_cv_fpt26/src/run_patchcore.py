"""Chay PatchCore 1 backbone: build bank (checkpoint) -> score holdout + test."""
import os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR, CKPT_DIR, ROOT, set_seed, holdout_split, read_test_csv
from features import iter_feats
from patchcore import Bank, Mahalanobis, greedy_coreset

def build_bank(files, model, long_side, dev, flips, bank_size, keep, bs, seed, jnorm=None):
    """Pass 1: gom patch (subsample tren tung anh) -> greedy coreset -> Bank."""
    pool, cls_all = [], []
    g = torch.Generator().manual_seed(seed)
    for fl in [None] + flips:
        for patch, cls in iter_feats(files, model, long_side, dev, fl, bs, jnorm=jnorm):
            B, N, D = patch.shape
            nk = max(1, int(N * keep))
            sel = torch.stack([torch.randperm(N, generator=g)[:nk] for _ in range(B)])
            pool.append(torch.gather(patch, 1, sel.unsqueeze(-1).expand(-1, -1, D)).reshape(-1, D))
            if fl is None: cls_all.append(cls)
    pool = torch.cat(pool)
    idx = greedy_coreset(pool, bank_size, dev, seed=seed)
    return Bank(pool[idx].contiguous(), dev), torch.cat(cls_all)

def score_files(bank, maha, files, model, long_side, dev, topq, bs, jnorm=None):
    ps, ms, tops, npatch = [], [], [], None
    for patch, cls in iter_feats(files, model, long_side, dev, None, bs, jnorm=jnorm):
        s_, t_, npatch = bank.score_batch(patch, topq)
        ps.append(s_); tops.append(t_); ms.append(maha.score(cls))
    return (torch.cat(ps).numpy(), torch.cat(ms).numpy(),
            torch.cat(tops).numpy().astype("float32"), npatch)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wrn50")
    ap.add_argument("--long", type=int, default=512)
    ap.add_argument("--splits", default="public")
    ap.add_argument("--flips", default="")
    ap.add_argument("--bank-size", type=int, default=30000)
    ap.add_argument("--keep", type=float, default=0.15, help="ti le patch giu lai moi anh truoc coreset")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--topq", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--cats", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jnorm", action="store_true", help="chuan hoa mien nen JPEG")
    a = ap.parse_args()
    set_seed()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flips = [f for f in a.flips.split(",") if f]
    cats = [c for c in (a.cats.split(",") if a.cats else CATS)]
    ck = os.path.join(CKPT_DIR, a.tag); os.makedirs(ck, exist_ok=True)
    res = {"_config": np.array([repr(vars(a))])}
    for cat in cats:
        t0 = time.time()
        fit_f, hold_f = holdout_split(cat)
        jn = ROOT if a.jnorm else None
        bank, cls = build_bank(fit_f, a.model, a.long, dev, flips, a.bank_size, a.keep, a.bs, 1337, jn)
        maha = Mahalanobis(cls)
        bank.save(os.path.join(ck, f"{cat}_bank.pt"))
        torch.save(maha.state(), os.path.join(ck, f"{cat}_maha.pt"))
        (res[f"{cat}/hold"], res[f"{cat}/hold_m"],
         res[f"{cat}/hold_top"], npz_np) = score_files(bank, maha, hold_f, a.model, a.long, dev, a.topq, a.bs, jn)
        res[f"{cat}/npatch"] = np.array([npz_np])
        for split in a.splits.split(","):
            rows = [r for r in read_test_csv(split) if r[1] == cat]
            p, m, tp, _ = score_files(bank, maha, [r[2] for r in rows], a.model, a.long, dev, a.topq, a.bs, jn)
            res[f"{cat}/{split}_ids"] = np.array([r[0] for r in rows])
            res[f"{cat}/{split}"], res[f"{cat}/{split}_m"] = p, m
            res[f"{cat}/{split}_top"] = tp
        h, t = res[f"{cat}/hold"], res[f"{cat}/{a.splits.split(',')[0]}"]
        print(f"  {cat}: bank={tuple(bank.bank.shape)} {time.time()-t0:.0f}s | "
              f"hold med={np.median(h):.3f} | test med={np.median(t):.3f} "
              f"p90={np.quantile(t,.9):.3f} | sep={(np.median(t)-np.median(h))/(h.std()+1e-9):+.2f}sd", flush=True)
    out = os.path.join(CACHE_DIR, f"scores_{a.tag}.npz")
    np.savez(out, **res); print("saved", out, "| ckpt", ck)

if __name__ == "__main__":
    main()
