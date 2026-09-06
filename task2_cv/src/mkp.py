"""Sinh submission private o gia tri p bat ky, tu score tho da luu. Khong chay lai suy luan."""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CKPT_DIR
from make_sub import write_sub
from variants import rank01

cfg = json.load(open('ckpt/BEST_81.2/config.json'))
raw = np.load(os.path.join(CKPT_DIR, 'rawscores_private_PRIVATE_SAFE.npz'), allow_pickle=True)
npatch = {}
for m in cfg["models"]:
    d = np.load(f"cache/scores_{m['tag']}.npz", allow_pickle=True)
    for c in CATS: npatch[(c, m['tag'])] = int(d[f"{c}/npatch"][0])

def scores(cat, topq=0.01):
    pt = []
    for bi, m in enumerate(cfg["models"]):
        n = npatch[(cat, m['tag'])]; k = max(1, int(round(n * topq)))
        t = raw[f"{cat}/{m['tag']}/top"][:, :k].mean(1)
        h = np.array(cfg['hold'][cat]['patch'][bi])
        mh = raw[f"{cat}/{m['tag']}/maha"]; hm = np.array(cfg['hold'][cat]['maha'][bi])
        for a, b in ((t, h), (mh, hm)):
            pt.append(rank01(np.concatenate([a, b]))[:len(a)])
    return raw[f"{cat}/ids"], np.mean(pt, 0)

if __name__ == "__main__":
    p = float(sys.argv[1]); name = sys.argv[2]
    topq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    lab = {}
    for c in CATS:
        ids, s = scores(c, topq)
        tau = np.quantile(s, 1 - p)
        for i, sid in enumerate(ids): lab[str(sid)] = int(s[i] >= tau)
    _, z = write_sub(lab, "private", name)
    print(f"  p={p}  topq={topq}  anom={sum(lab.values())}/{len(lab)} ({sum(lab.values())/len(lab):.1%})  -> {z}")
