"""Inference-only: load saved fold checkpoints and predict a test set.

Used for the private round so no retraining is required. `id` is read only to
join predictions to rows.
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from common import *
from train_mtl import MTL, DS, collate

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--test", default="private_test")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=256)
a = ap.parse_args()

set_seed()
te = load_test(a.test)
cks = sorted(glob.glob(f"ckpt/{a.tag}/fold*.pt"))
assert cks, f"no checkpoints in ckpt/{a.tag}"
targs = torch.load(cks[0], map_location="cpu", weights_only=False)["args"]
tok = AutoTokenizer.from_pretrained(targs["model"])

if targs["view"] == "dual":
    from normalize import build as build_norm
    nz, _ = build_norm(load_labeled())
    view = [(t, nz(t)) for t in te.text]
else:
    view = list(te.text)

dl = DataLoader(DS(view, tok, targs["maxlen"]), batch_size=a.bs,
                collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=4)
H, N = [], []
for c in cks:
    st = torch.load(c, map_location="cpu", weights_only=False)
    model = MTL(targs["model"]).cuda(); model.load_state_dict(st["model"]); model.eval()
    h, n = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            lh, ln = model(b["input_ids"].cuda(), b["attention_mask"].cuda())
            h.append(lh.float().softmax(-1).cpu()); n.append(ln.float().softmax(-1).cpu())
    H.append(torch.cat(h).numpy()); N.append(torch.cat(n).numpy())
    print(f"  {os.path.basename(c)}  fold={st['fold']} ep={st['epoch']} cv={st['score']:.4f}")
    del model; torch.cuda.empty_cache()

np.savez(a.out, te_h=np.mean(H, 0), te_n=np.mean(N, 0))
print(f"[saved] {a.out}  n={len(te)}  from {len(cks)} checkpoints")
