"""Re-tune the noise bias under the *estimated test* noise prior instead of the
train prior. Sample weights turn the OOF into a surrogate for the test mix.
The prior itself is an estimate, so this is offered as an alternative
submission to be settled on the leaderboard, not asserted as correct.
"""
import sys, os; sys.path.insert(0, "src")
import numpy as np
from common import *
from sklearn.metrics import f1_score

set_seed(); df = load_labeled(); df["fold"] = make_folds(df)
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--npz",default="oof/visobert_ep_oof.npz"); ap.add_argument("--out",default="oof/bias_prior.npz"); A=ap.parse_args()
d = np.load(A.npz); m = d["seen"]
y = df.y_noise.values[m]; P = d["n"][m]
train_p = np.bincount(y, minlength=7) / len(y)
test_p = np.array([0.25] + [0.125] * 6)          # from four surface statistics
w = (test_p / train_p)[y]
print("train prior:", np.round(train_p, 3))
print("test  prior:", np.round(test_p, 3), "(estimated)")

def wf1(yt, yp, sw):
    return f1_score(yt, yp, average="macro", labels=range(7), zero_division=0, sample_weight=sw)

L = np.log(np.clip(P, 1e-9, 1)); b = np.zeros(7)
best = wf1(y, L.argmax(1), w)
for _ in range(6):
    imp = False
    for c in range(7):
        cur, bs, bb = b[c], best, b[c]
        for g in np.linspace(-2, 2, 41):
            b[c] = g; s = wf1(y, (L + b).argmax(1), w)
            if s > bs: bs, bb = s, g
        b[c] = bb
        if bs > best + 1e-9: best, imp = bs, True
    if not imp: break
print(f"prior-reweighted macro-F1: argmax={wf1(y,L.argmax(1),w):.4f} -> tuned={best:.4f}")
print("bias:", np.round(b, 2))
np.savez(A.out, bias_h=np.zeros(3), bias_n=b); print("[saved]",A.out)
