"""Per-class log-bias tuning to maximise macro-F1 directly.

argmax of the raw posterior does NOT maximise macro-F1 under class imbalance.
We add a per-class additive bias in log space and optimise it by coordinate
ascent. Honesty guard: the bias is fitted on one half of the OOF rows and
scored on the held-out half, so the reported gain is not the fitted gain.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import *


def coord_ascent(P, y, n_cls, rounds=6, grid=np.linspace(-2.0, 2.0, 41)):
    b = np.zeros(n_cls)
    L = np.log(np.clip(P, 1e-9, 1))
    best = macro_f1(y, (L + b).argmax(1), n_cls)
    for _ in range(rounds):
        improved = False
        for c in range(n_cls):
            cur = b[c]; bb, bs = cur, best
            for g in grid:
                b[c] = g
                s = macro_f1(y, (L + b).argmax(1), n_cls)
                if s > bs:
                    bs, bb = s, g
            b[c] = bb
            if bs > best + 1e-9:
                best, improved = bs, True
        if not improved:
            break
    return b, best


def evaluate(name, P, y, n_cls, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)); a_, b_ = idx[::2], idx[1::2]
    base = macro_f1(y, P.argmax(1), n_cls)
    # honest: fit on half A, score on half B (and vice-versa)
    ba, _ = coord_ascent(P[a_], y[a_], n_cls)
    bb, _ = coord_ascent(P[b_], y[b_], n_cls)
    L = np.log(np.clip(P, 1e-9, 1))
    hb = 0.5 * (macro_f1(y[b_], (L[b_] + ba).argmax(1), n_cls) +
                macro_f1(y[a_], (L[a_] + bb).argmax(1), n_cls))
    bfull, sfull = coord_ascent(P, y, n_cls)
    print(f"  {name:6s} argmax={base:.4f}   held-out tuned={hb:.4f} ({hb-base:+.4f})   "
          f"full-fit={sfull:.4f} (optimistic)")
    print(f"         bias = {np.round(bfull, 2)}")
    return bfull, hb - base


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True)
    ap.add_argument("--save", default=None); a = ap.parse_args()
    set_seed(); df = load_labeled(); df["fold"] = make_folds(df)
    d = np.load(a.npz); m = d["seen"]
    print(f"[tune] {a.npz}  n={m.sum()}")
    bh, gh = evaluate("hate", d["h"][m], df.y_hate.values[m], 3)
    bn, gn = evaluate("noise", d["n"][m], df.y_noise.values[m], 7)
    s0, f0h, f0n = official_score(df.y_hate.values[m], d["h"][m].argmax(1),
                                  df.y_noise.values[m], d["n"][m].argmax(1))
    print(f"\n  OFFICIAL argmax = {s0:.4f}")
    print(f"  OFFICIAL tuned  ~ {s0 + 0.85*gh + 0.15*gn:.4f}   ({0.85*gh + 0.15*gn:+.4f})")
    if a.save:
        np.savez(a.save, bias_h=bh, bias_n=bn); print(f"  [saved] {a.save}")
