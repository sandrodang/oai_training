"""Quet topq / kieu gop OFFLINE tu top-512 khoang cach patch da luu. Ton 0 luot nop.
Xep hang cac lua chon bang AUC_proxy (test vs holdout-normal)."""
import os, sys, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR
from proxy import auc

def pool(top, npatch, topq, mode="mean"):
    """top: [n_img, 512] khoang cach patch giam dan."""
    k = max(1, int(round(npatch * topq)))
    k = min(k, top.shape[1])
    v = top[:, :k]
    return {"mean": v.mean(1), "max": v[:, 0], "median": np.median(v, 1)}[mode]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="public")
    a = ap.parse_args()
    d = np.load(os.path.join(CACHE_DIR, f"scores_{a.tag}.npz"), allow_pickle=True)
    if f"{CATS[0]}/hold_top" not in d:
        sys.exit(f"{a.tag} chua luu top-K patch distance -> chay lai run_patchcore.py")
    grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
    print(f"{'topq':>7s} {'mode':>7s} {'mean':>7s} " + " ".join(f"{c[-2:]:>6s}" for c in CATS))
    rows = []
    for q in grid:
        for mode in ("mean", "max"):
            per = {}
            for c in CATS:
                n = int(d[f"{c}/npatch"][0])
                per[c] = auc(pool(d[f"{c}/{a.split}_top"], n, q, mode),
                             pool(d[f"{c}/hold_top"], n, q, mode))
            rows.append((q, mode, float(np.mean(list(per.values()))), per))
    for q, mode, m, per in sorted(rows, key=lambda r: -r[2])[:12]:
        print(f"{q:7.4f} {mode:>7s} {m:7.4f} " + " ".join(f"{per[c]:6.4f}" for c in CATS))
    print("\nTot nhat TUNG CATEGORY:")
    for c in CATS:
        b = max(rows, key=lambda r: r[3][c])
        print(f"  {c}: topq={b[0]:.4f} mode={b[1]:>6s} auc_proxy={b[3][c]:.4f}")

if __name__ == "__main__":
    main()
