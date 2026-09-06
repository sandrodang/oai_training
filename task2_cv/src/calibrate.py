"""Hieu chuan bo uoc luong BA bang MOT con so LB da biet, roi doc BA tung category MIEN PHI.

Ta biet: cau hinh X o nguong quantile p cho macro-BA = B (do LB tra ve).
Bo uoc luong khu chap cho BA_c(tau) voi tham so duy nhat pi.
=> chon pi sao cho mean_c BA_c(tau_p) = B. Sau do BA_c va tau_c toi uu la mien phi.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS
from submit import get_scores
from deconv import ba_curve

def predict_ba(sc, pi, p=None, per_cat_tau=None, ngrid=600):
    out = {}
    for c, (_, te, ho) in sc.items():
        lo, hi = min(te.min(), ho.min()), max(te.max(), ho.max())
        grid = np.linspace(lo, hi, ngrid)
        ba, tpr, tnr = ba_curve(te, ho, grid, pi)
        if per_cat_tau is not None:
            i = int(np.argmax(ba)); out[c] = (ba[i], grid[i], float((te >= grid[i]).mean()))
        else:
            t = np.quantile(te, 1 - p)
            i = int(np.argmin(np.abs(grid - t)))
            out[c] = (ba[i], t, p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--maha", action="store_true")
    ap.add_argument("--p", type=float, required=True, help="p cua lan nop da biet diem")
    ap.add_argument("--lb", type=float, required=True, help="macro-BA thuc te LB tra ve (0-1)")
    a = ap.parse_args()
    sc = get_scores(a.tags.split(","), "public", a.maha)

    best = None
    for pi in np.arange(0.10, 0.90, 0.005):
        m = np.mean([v[0] for v in predict_ba(sc, pi, p=a.p).values()])
        if best is None or abs(m - a.lb) < abs(best[1] - a.lb): best = (pi, m)
    pi = best[0]
    print(f"pi hieu chuan = {pi:.3f}  (BA du bao {best[1]:.4f} vs LB thuc te {a.lb:.4f})")

    cur = predict_ba(sc, pi, p=a.p)
    opt = predict_ba(sc, pi, per_cat_tau=True)
    print(f"\n{'cat':13s} {'BA @p chung':>12s} {'BA @tau rieng':>14s} {'loi':>6s} {'p_c toi uu':>11s}")
    for c in CATS:
        print(f"{c:13s} {cur[c][0]:12.4f} {opt[c][0]:14.4f} {opt[c][0]-cur[c][0]:+6.4f} {opt[c][2]:11.3f}")
    mc, mo = np.mean([cur[c][0] for c in CATS]), np.mean([opt[c][0] for c in CATS])
    print(f"{'TRUNG BINH':13s} {mc:12.4f} {mo:14.4f} {mo-mc:+6.4f}")
    print(f"\n=> nguong rieng tung category du bao them {100*(mo-mc):+.2f} diem")
    print("   p_c de dung:", {c: round(opt[c][2], 3) for c in CATS})

if __name__ == "__main__":
    main()
