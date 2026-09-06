"""Toi uu ensemble OFFLINE tu top-K patch distance da luu. Khong ton luot nop.
Quet: topq rieng tung nhanh / bo nhanh maha trung lap / rank vs z-score.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR
from proxy import auc

_D = {}
def load(tag):
    if tag not in _D:
        _D[tag] = np.load(os.path.join(CACHE_DIR, f"scores_{tag}.npz"), allow_pickle=True)
    return _D[tag]

def pooled(tag, cat, split, topq, mode="mean"):
    d = load(tag); n = int(d[f"{cat}/npatch"][0])
    k = min(max(1, int(round(n * topq))), d[f"{cat}/{split}_top"].shape[1])
    f = lambda a: (a[:, :k].mean(1) if mode == "mean" else a[:, 0])
    return f(d[f"{cat}/{split}_top"]), f(d[f"{cat}/hold_top"])

def rank01(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1)

def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def ens(cat, branches, split="public", norm="rank"):
    """branches: list (tag, topq, mode, use_maha)."""
    te, ho = [], []
    g = rank01 if norm == "rank" else z
    for tag, topq, mode, um in branches:
        t, h = pooled(tag, cat, split, topq, mode)
        j = g(np.concatenate([t, h])); te.append(j[:len(t)]); ho.append(j[len(t):])
        if um:
            d = load(tag)
            j = g(np.concatenate([d[f"{cat}/{split}_m"], d[f"{cat}/hold_m"]]))
            te.append(j[:len(t)]); ho.append(j[len(t):])
    return np.mean(te, 0), np.mean(ho, 0)

def score_cfg(branches, norm="rank"):
    per = {c: auc(*ens(c, branches, "public", norm)) for c in CATS}
    return float(np.mean(list(per.values()))), per
