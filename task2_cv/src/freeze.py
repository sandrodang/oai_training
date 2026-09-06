"""Dong bang cau hinh cuoi: ghi config.json gom models, rule, tham so, va
phan bo score cua train-normal holdout (de tai lap nguong o private)."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR, CKPT_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="tag:model:long[:jnorm], phay ngan cach. vd wrn50_L512:wrn50:512,dinov2b_L770:dinov2b:770")
    ap.add_argument("--rule", default="p", choices=["p", "hybrid", "deconv"])
    ap.add_argument("--p", type=float, default=0.44)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--pi", type=float, default=0.39)
    ap.add_argument("--topq", type=float, default=0.01)
    ap.add_argument("--maha", action="store_true")
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "FINAL", "config.json"))
    a = ap.parse_args()

    models, hold = [], {c: {"patch": [], "maha": []} for c in CATS}
    for spec in a.models.split(","):
        parts = spec.split(":")
        tag, mdl, lng = parts[0], parts[1], int(parts[2])
        e = {"tag": tag, "model": mdl, "long": lng,
             "jnorm": len(parts) > 3 and parts[3] == "jnorm",
             # PHAI khop batch size cua lan chay sinh ra checkpoint:
             # bf16 tich luy theo batch khac nhau -> lech nho -> lat nhan sat nguong
             "bs": int(parts[4]) if len(parts) > 4 else
                   (2 if (lng > 800 or mdl == "dinov2g_ml") else (8 if mdl == "wrn50" else 4))}
        if mdl == "dinomaly":
            e.update({"layers": [6, 8, 10, 12], "d_model": 512, "nblk": 4, "enc_bs": 4})
        models.append(e)
        d = np.load(os.path.join(CACHE_DIR, f"scores_{tag}.npz"), allow_pickle=True)
        for c in CATS:
            hold[c]["patch"].append(d[f"{c}/hold"].tolist())
            hold[c]["maha"].append(d[f"{c}/hold_m"].tolist())

    q = {}
    if a.rule == "hybrid":
        from submit import get_scores
        sc = get_scores([m["tag"] for m in models], "public", a.maha)
        for c, (_, te, ho) in sc.items():
            q[c] = float((ho >= np.quantile(te, 1 - a.p)).mean())   # suy ra q_c MIEN PHI tu p*

    cfg = {"models": models, "rule": a.rule, "p": a.p, "w": a.w, "pi": a.pi,
           "q": q, "topq": a.topq, "maha": a.maha, "hold": hold, "seed": 1337}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(cfg, open(a.out, "w"))
    print("freeze ->", a.out)
    print("  models:", [m["tag"] for m in models], "| rule:", a.rule,
          "| p:", a.p, "| maha:", a.maha)
    if q: print("  q_c:", {k: round(v, 3) for k, v in q.items()})

if __name__ == "__main__":
    main()
