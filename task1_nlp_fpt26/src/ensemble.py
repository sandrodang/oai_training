"""Weighted probability ensemble with weights fitted on OOF.

Honesty guard: weights (and the noise bias) are fitted on one half of the OOF
rows and scored on the held-out half, so the reported number is not the
fitted number. The final artefact refits on all OOF rows.
"""
import sys, os, argparse, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import *
from tune_bias import coord_ascent  # NOTE: train-prior tuner. For submissions use tune_prior.py (test-prior) instead -- see v8 regression

ap = argparse.ArgumentParser()
ap.add_argument("--npz", nargs="+", required=True)
ap.add_argument("--out", default="oof/ensemble.npz")
ap.add_argument("--bias_noise", action="store_true", default=True)
a = ap.parse_args()

set_seed(); df = load_labeled(); df["fold"] = make_folds(df)
D = [np.load(f) for f in a.npz]
m = D[0]["seen"].copy()
for d in D[1:]:
    m &= d["seen"]
print(f"[ens] {len(D)} models, {m.sum()} common OOF rows")
yh, yn = df.y_hate.values[m], df.y_noise.values[m]
Hs = [d["h"][m] for d in D]; Ns = [d["n"][m] for d in D]
for f, H, N in zip(a.npz, Hs, Ns):
    print(f"   {os.path.basename(f):28s} hate={macro_f1(yh,H.argmax(1),3):.4f} "
          f"noise={macro_f1(yn,N.argmax(1),7):.4f}")


def search_w(Ps, y, n_cls, steps=11):
    """grid over the simplex (coarse but exhaustive for <=4 models)"""
    k = len(Ps); best, bw = -1, None
    for combo in itertools.product(range(steps + 1), repeat=k - 1):
        if sum(combo) > steps:
            continue
        w = np.array(list(combo) + [steps - sum(combo)], float) / steps
        P = sum(wi * p for wi, p in zip(w, Ps))
        s = macro_f1(y, P.argmax(1), n_cls)
        if s > best:
            best, bw = s, w
    return bw, best


rng = np.random.RandomState(0); idx = rng.permutation(m.sum())
A, B = idx[::2], idx[1::2]
report = {}
for name, Ps, y, n_cls, use_bias in [("hate", Hs, yh, 3, False), ("noise", Ns, yn, 7, True)]:
    single = max(macro_f1(y, P.argmax(1), n_cls) for P in Ps)
    hold = []
    for fit, ev in ((A, B), (B, A)):
        w, _ = search_w([P[fit] for P in Ps], y[fit], n_cls)
        P_ev = sum(wi * p[ev] for wi, p in zip(w, Ps))
        if use_bias:
            b, _ = coord_ascent(sum(wi * p[fit] for wi, p in zip(w, Ps)), y[fit], n_cls)
            hold.append(macro_f1(y[ev], (np.log(np.clip(P_ev, 1e-9, 1)) + b).argmax(1), n_cls))
        else:
            hold.append(macro_f1(y[ev], P_ev.argmax(1), n_cls))
    honest = float(np.mean(hold))
    w_full, s_full = search_w(Ps, y, n_cls)
    P_full = sum(wi * p for wi, p in zip(w_full, Ps))
    b_full = np.zeros(n_cls)
    if use_bias:
        b_full, s_full = coord_ascent(P_full, y, n_cls)
    print(f"\n  {name}: best single={single:.4f}  ensemble held-out={honest:.4f} "
          f"({honest-single:+.4f})  [full-fit {s_full:.4f}]")
    print(f"         weights={np.round(w_full,3)}  bias={np.round(b_full,2)}")
    report[name] = (w_full, b_full, honest)

wh, bh, hh = report["hate"]; wn, bn, hn = report["noise"]
print(f"\n  ESTIMATED OFFICIAL (held-out) = {0.85*hh + 0.15*hn:.4f}")
te_h = sum(wi * d["te_h"] for wi, d in zip(wh, D))
te_n = sum(wi * d["te_n"] for wi, d in zip(wn, D))
np.savez(a.out, te_h=te_h * np.exp(bh), te_n=te_n * np.exp(bn),
         h=sum(wi * d["h"] for wi, d in zip(wh, D)) * np.exp(bh),
         n=sum(wi * d["n"] for wi, d in zip(wn, D)) * np.exp(bn), seen=m)
print(f"  [saved] {a.out}")
