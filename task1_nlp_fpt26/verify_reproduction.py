"""Post-verification: regenerate a submitted CSV from the packaged checkpoints
and compare it to the file that was actually submitted.

    ./.venv/bin/python verify_reproduction.py --tag visobert_fgm \
        --sub sub/v7_visobert_fgm/task1_public_output.csv --bias oof/bias_fgm.npz

Exits 0 only on an exact, row-for-row match.
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from common import *
from train_mtl import MTL, DS, collate

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--sub", required=True)
ap.add_argument("--bias", default=None)
ap.add_argument("--test", default="public_test")
a = ap.parse_args()

set_seed()
te = load_test(a.test)
cks = sorted(glob.glob(f"ckpt/{a.tag}/fold*.pt"))
assert cks, f"no checkpoints under ckpt/{a.tag}"
targs = torch.load(cks[0], map_location="cpu", weights_only=False)["args"]
tok = AutoTokenizer.from_pretrained(targs["model"])
dl = DataLoader(DS(list(te.text), tok, targs["maxlen"]), batch_size=256,
                collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=4)

H, N = [], []
for c in cks:
    m = MTL(targs["model"]).cuda()
    m.load_state_dict(torch.load(c, map_location="cpu", weights_only=False)["model"]); m.eval()
    h, n = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            lh, ln = m(b["input_ids"].cuda(), b["attention_mask"].cuda())
            h.append(lh.float().softmax(-1).cpu()); n.append(ln.float().softmax(-1).cpu())
    H.append(torch.cat(h).numpy()); N.append(torch.cat(n).numpy())
    del m; torch.cuda.empty_cache()

bh = np.zeros(3); bn = np.zeros(7)
if a.bias:
    b = np.load(a.bias); bh, bn = b["bias_h"], b["bias_n"]
Hm = np.log(np.clip(np.mean(H, 0), 1e-9, 1)) + bh
Nm = np.log(np.clip(np.mean(N, 0), 1e-9, 1)) + bn
rep = pd.DataFrame({"id": te["id"],
                    "pred_label": [HATE_LABELS[i] for i in Hm.argmax(1)],
                    "pred_noise_type": [NOISE_LABELS[i] for i in Nm.argmax(1)]})
sub = pd.read_csv(a.sub)
same = (rep.id.values == sub.id.values).all() \
    and (rep.pred_label.values == sub.pred_label.values).all() \
    and (rep.pred_noise_type.values == sub.pred_noise_type.values).all()
print(f"rows={len(rep)}  checkpoints={len(cks)}")
print(f"pred_label      match: {(rep.pred_label.values==sub.pred_label.values).mean()*100:.2f}%")
print(f"pred_noise_type match: {(rep.pred_noise_type.values==sub.pred_noise_type.values).mean()*100:.2f}%")
print("VERDICT:", "EXACT REPRODUCTION" if same else "MISMATCH")
sys.exit(0 if same else 1)
