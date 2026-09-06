"""Nguong hybrid + sinh submission.
tau(c) = w*Q_{1-p}(score_test) + (1-w)*Q_{1-q_c}(score_holdout_normal)
- p: tham so duy nhat sweep tren public LB
- q_c: suy ra MIEN PHI tu p* (khong ton luot nop) bang derive_q.py
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR, CKPT_DIR
from make_sub import write_sub

def load(tag):
    return np.load(os.path.join(CACHE_DIR, f"scores_{tag}.npz"), allow_pickle=True)

def rank01(x):
    """rank-normalize ve [0,1] trong noi bo category -> ensemble khong can calibrate scale."""
    r = np.empty(len(x), float); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1)

def repool(d, cat, split, topq):
    """Tinh lai diem anh tu top-K khoang cach patch da luu (topq nho -> max pooling)."""
    n = int(d[f"{cat}/npatch"][0])
    k = min(max(1, int(round(n * topq))), d[f"{cat}/{split}_top"].shape[1])
    return d[f"{cat}/{split}_top"][:, :k].mean(1), d[f"{cat}/hold_top"][:, :k].mean(1)

def get_scores(tags, split, use_maha=False, topq=None):
    """-> cat -> (ids, test_score, hold_score). Ensemble = trung binh RANK trong category."""
    ds = [load(t) for t in tags]
    out = {}
    for cat in CATS:
        ids = ds[0][f"{cat}/{split}_ids"]
        te, ho = [], []
        for d in ds:
            assert list(d[f"{cat}/{split}_ids"]) == list(ids), "thu tu id lech giua cac tag"
            assert len(d[f"{cat}/hold"]) == len(ds[0][f"{cat}/hold"]), \
                f"holdout lech giua cac tag o {cat} - phai chay lai cho dong bo"
            t, h = (repool(d, cat, split, topq) if topq is not None
                    else (d[f"{cat}/{split}"], d[f"{cat}/hold"]))
            n = len(h)
            j = np.concatenate([t, h]); j = rank01(j)      # rank chung test+holdout
            te.append(j[:len(t)]); ho.append(j[len(t):])
            if use_maha:
                jm = rank01(np.concatenate([d[f"{cat}/{split}_m"], d[f"{cat}/hold_m"]]))
                te.append(jm[:len(t)]); ho.append(jm[len(t):])
        out[cat] = (ids, np.mean(te, 0), np.mean(ho, 0))
    return out

def taus(sc, p, q=None, w=1.0, pc=None):
    t = {}
    for cat, (_, te, ho) in sc.items():
        pp = float(pc[cat]) if pc else p
        a = np.quantile(te, 1 - pp)
        if q is None or w >= 1.0:
            t[cat] = a
        else:
            b = np.quantile(ho, 1 - float(q[cat]))
            t[cat] = w * a + (1 - w) * b
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="phay ngan cach, vd wrn50_L512,dinov2b_L518")
    ap.add_argument("--split", default="public", choices=["public", "private"])
    ap.add_argument("--p", type=float, default=0.35)
    ap.add_argument("--w", type=float, default=1.0, help="1.0 = chi neo test-quantile; 0.5 = hybrid")
    ap.add_argument("--q-file", default=None)
    ap.add_argument("--maha", action="store_true")
    ap.add_argument("--pc", default=None, help="JSON p rieng tung category")
    ap.add_argument("--shrink", type=float, default=1.0, help="co ve p chung: 1=dung nguyen, 0=bo qua")
    ap.add_argument("--clamp", default="0.35,0.60")
    ap.add_argument("--topq", type=float, default=None, help="gop lai tu top-K da luu; 0.0001 = max pooling")
    ap.add_argument("--name", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    tags = a.tags.split(",")
    sc = get_scores(tags, a.split, a.maha, a.topq)
    q = json.load(open(a.q_file)) if a.q_file else None
    pc = None
    if a.pc:
        raw = json.loads(a.pc) if a.pc.strip().startswith("{") else json.load(open(a.pc))
        lo, hi = [float(x) for x in a.clamp.split(",")]
        pc = {c: float(np.clip(a.p + a.shrink * (np.clip(v, lo, hi) - a.p), lo, hi))
              for c, v in raw.items()}
        print("p_c dung that:", {c: round(v, 3) for c, v in pc.items()})
    T = taus(sc, a.p, q, a.w, pc)

    if a.report:
        print(f"{'cat':13s} {'hold q50':>9s} {'test q50':>9s} {'t q75':>7s} {'t q90':>7s} "
              f"{'tau':>7s} {'#anom':>6s} {'q_c ngu y':>10s}")
        for cat, (_, te, ho) in sc.items():
            qc = (ho >= T[cat]).mean()
            print(f"{cat:13s} {np.median(ho):9.4f} {np.median(te):9.4f} {np.quantile(te,.75):7.4f} "
                  f"{np.quantile(te,.90):7.4f} {T[cat]:7.4f} {int((te>=T[cat]).sum()):6d} {qc:10.3f}")

    labels = {}
    for cat, (ids, te, _) in sc.items():
        for i, sid in enumerate(ids):
            labels[str(sid)] = int(te[i] >= T[cat])
    c, z = write_sub(labels, a.split, a.name)
    meta = {"tags": tags, "p": a.p, "w": a.w, "maha": a.maha, "pc": pc, "shrink": a.shrink, "topq": a.topq,
            "q": q, "taus": {k: float(v) for k, v in T.items()},
            "n_anom": {cat: int((te >= T[cat]).sum()) for cat, (_, te, _) in sc.items()}}
    json.dump(meta, open(os.path.join(os.path.dirname(c), "config.json"), "w"), indent=1)
    print("\nCSV:", c); print("ZIP:", z)

if __name__ == "__main__":
    main()
