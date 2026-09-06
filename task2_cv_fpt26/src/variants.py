"""Sinh nhieu bien the submission tu score tho da luu — KHONG chay lai suy luan.
Cac bien the chi khac buoc gop diem cuoi, dung chung model va dac trung."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CKPT_DIR
from make_sub import write_sub

def rank01(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1)
def z(x): return (x - x.mean()) / (x.std() + 1e-9)

def build(raw, cfg, topq, norm, maha=True, npatch=None):
    out = {}
    g = rank01 if norm == "rank" else z
    for cat in CATS:
        ids = raw[f"{cat}/ids"]; parts = []
        for m in cfg["models"]:
            top = raw[f"{cat}/{m['tag']}/top"]
            n = npatch.get((cat, m["tag"]), top.shape[1] * 100)
            k = min(max(1, int(round(n * topq))), top.shape[1])
            parts.append(g(top[:, :k].mean(1)))
            if maha: parts.append(g(raw[f"{cat}/{m['tag']}/maha"]))
        s = np.mean(parts, 0)
        tau = np.quantile(s, 1 - cfg["p"])
        for i, sid in enumerate(ids): out[str(sid)] = int(s[i] >= tau)
    return out

if __name__ == "__main__":
    split, name = sys.argv[1], sys.argv[2]
    cfg = json.load(open(sys.argv[3]))
    raw = np.load(os.path.join(CKPT_DIR, f"rawscores_{split}_{name}.npz"), allow_pickle=True)
    # so patch that lay tu cache public cua tung nhanh
    npatch = {}
    for m in cfg["models"]:
        f = f"cache/scores_{m['tag']}.npz"
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            for c in CATS:
                if f"{c}/npatch" in d: npatch[(c, m["tag"])] = int(d[f"{c}/npatch"][0])
    V = [("V1_top1pct_rank", 0.01,   "rank", True),
         ("V2_max_rank",     0.0001, "rank", True),
         ("V3_top1pct_z",    0.01,   "z",    True),
         ("V4_max_z",        0.0001, "z",    True),
         ("V5_top05pct_rank",0.005,  "rank", True)]
    base = None
    for vn, q, nm, mh in V:
        lab = build(raw, cfg, q, nm, mh, npatch)
        c, zp = write_sub(lab, split, vn)
        if base is None: base = lab
        d = sum(base[k] != lab[k] for k in base)
        print(f"  {vn:18s} anom={sum(lab.values()):4d}/{len(lab)}  khac V1: {d:3d} nhan  -> {zp}")
