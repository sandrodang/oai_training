"""Pipeline FREEZE end-to-end: load checkpoint -> score -> nguong -> submission.
Dung cho ca public va private. Khong hoc gi tu tap test.

    python src/predict.py --config ckpt/FINAL/config.json --split private --name final_pri
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CKPT_DIR, ROOT, set_seed, read_test_csv
from features import iter_feats
from patchcore import Bank
from dinomaly import Decoder, encode as dino_encode, score as dino_score
from make_sub import write_sub
from submit import rank01

def load_maha(path):
    d = torch.load(path, map_location="cpu")
    class M:
        mu, P = d["mu"], d["P"]
        def score(self, X):
            Xc = X.double() - self.mu
            return ((Xc @ self.P) * Xc).sum(1).clamp(min=0).sqrt().float()
    return M()

def score_split(cfg, cat, files, dev):
    """-> (patch_rank_inputs, maha_rank_inputs) : list score tho tung branch."""
    ps, ms, tops = [], [], []
    for m in cfg["models"]:
        ck = os.path.join(CKPT_DIR, m["tag"])
        if m["model"] == "dinomaly":
            # nhanh tai tao dac trung: nap decoder da huan luyen tren train-normal
            feat = dino_encode(files, m["long"], dev, m.get("enc_bs", 4), m["layers"])
            dec = Decoder(feat.shape[-1], m["d_model"], m["nblk"], 0.0).to(dev)
            dec.load_state_dict(torch.load(os.path.join(ck, f"{cat}_dec.pt"), map_location="cpu"))
            sc, tp, _ = dino_score(dec, feat, dev, topq=cfg["topq"])
            ps.append(sc); ms.append(sc); tops.append(tp)
            del dec, feat; torch.cuda.empty_cache()
            continue
        bank = Bank.load(os.path.join(ck, f"{cat}_bank.pt"), dev)
        maha = load_maha(os.path.join(ck, f"{cat}_maha.pt"))
        jn = ROOT if m.get("jnorm") else None
        p_, m_, t_ = [], [], []
        for patch, cls in iter_feats(files, m["model"], m["long"], dev, None,
                                     m.get("bs", 4), jnorm=jn):
            s_, tp_, _ = bank.score_batch(patch, cfg["topq"])
            p_.append(s_); t_.append(tp_); m_.append(maha.score(cls))
        ps.append(torch.cat(p_).numpy()); ms.append(torch.cat(m_).numpy())
        tops.append(torch.cat(t_).numpy().astype("float32"))
        del bank; torch.cuda.empty_cache()
    return ps, ms, tops

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, choices=["public", "private"])
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    set_seed()
    cfg = json.load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, rep, raw = {}, [], {}
    for cat in CATS:
        rows = [r for r in read_test_csv(a.split) if r[1] == cat]
        ids = [r[0] for r in rows]
        ps, ms, tops = score_split(cfg, cat, [r[2] for r in rows], dev)
        raw[f"{cat}/ids"] = np.array(ids)
        for bi, m in enumerate(cfg["models"]):
            raw[f"{cat}/{m['tag']}/top"] = tops[bi]
            raw[f"{cat}/{m['tag']}/maha"] = ms[bi]
        # phan bo normal da FREEZE tu train-normal holdout (khong dung test)
        hp = [np.array(x) for x in cfg["hold"][cat]["patch"]]
        hm = [np.array(x) for x in cfg["hold"][cat]["maha"]]
        te, ho = [], []
        for a_, h_ in list(zip(ps, hp)) + (list(zip(ms, hm)) if cfg["maha"] else []):
            j = rank01(np.concatenate([a_, h_]))
            te.append(j[:len(a_)]); ho.append(j[len(a_):])
        te, ho = np.mean(te, 0), np.mean(ho, 0)

        rule = cfg["rule"]
        if rule == "p":
            tau = np.quantile(te, 1 - cfg["p"])
        elif rule == "hybrid":
            tau = cfg["w"] * np.quantile(te, 1 - cfg["p"]) + \
                  (1 - cfg["w"]) * np.quantile(ho, 1 - cfg["q"][cat])
        elif rule == "deconv":
            from deconv import best_tau      # nhanh khong dung o cau hinh cuoi
            tau, _, _, _ = best_tau(te, ho, pi=cfg["pi"])
        else:
            raise ValueError(rule)
        for i, sid in enumerate(ids):
            labels[sid] = int(te[i] >= tau)
        rep.append((cat, float(tau), int((te >= tau).sum()), len(ids)))
        print(f"  {cat}: tau={tau:.4f}  anom={rep[-1][2]}/{len(ids)} ({rep[-1][2]/len(ids):.1%})", flush=True)

    np.savez(os.path.join(CKPT_DIR, f"rawscores_{a.split}_{a.name}.npz"), **raw)
    c, z = write_sub(labels, a.split, a.name)
    json.dump({"config": cfg, "report": rep},
              open(os.path.join(os.path.dirname(c), "run_meta.json"), "w"), indent=1)
    print("\nCSV:", c); print("ZIP:", z)

if __name__ == "__main__":
    main()
