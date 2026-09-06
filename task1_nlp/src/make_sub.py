"""Assemble a submission from saved probability files.
`id` is read only to join predictions to rows -- never used as a feature."""
import sys, os, argparse, zipfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from common import *

p = argparse.ArgumentParser()
p.add_argument("--npz", nargs="+", required=True)
p.add_argument("--w", nargs="+", type=float, default=None)
p.add_argument("--test", default="public_test")
p.add_argument("--phase", default="public")
p.add_argument("--name", default="FPTU_Promt_Engineer")
p.add_argument("--ver", required=True)
p.add_argument("--bias_npz", default=None)
p.add_argument("--bias_h", nargs=3, type=float, default=[0, 0, 0])
p.add_argument("--bias_n", nargs=7, type=float, default=[0] * 7)
a = p.parse_args()

w = a.w or [1.0] * len(a.npz)
w = np.array(w, float); w /= w.sum()
H = N = None
for wi, f in zip(w, a.npz):
    d = np.load(f)
    H = wi * d["te_h"] if H is None else H + wi * d["te_h"]
    N = wi * d["te_n"] if N is None else N + wi * d["te_n"]
if a.bias_npz:
    b = np.load(a.bias_npz); a.bias_h = list(b["bias_h"]); a.bias_n = list(b["bias_n"])
H = np.log(np.clip(H,1e-9,1)) + np.array(a.bias_h); N = np.log(np.clip(N,1e-9,1)) + np.array(a.bias_n)

te = load_test(a.test)
assert len(te) == len(H), f"row mismatch {len(te)} vs {len(H)}"
sub = pd.DataFrame({"id": te["id"],
                    "pred_label":      [HATE_LABELS[i] for i in H.argmax(1)],
                    "pred_noise_type": [NOISE_LABELS[i] for i in N.argmax(1)]})
os.makedirs(f"sub/{a.ver}", exist_ok=True)
csv = f"sub/{a.ver}/task1_{a.phase}_output.csv"
sub.to_csv(csv, index=False, encoding="utf-8")
z = f"sub/{a.ver}/{a.name}_task1_{'pub' if a.phase=='public' else 'pri'}.zip"
if os.path.exists(z): os.remove(z)
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    f.write(csv, os.path.basename(csv))
print(f"[write] {z}  rows={len(sub)}")
print(sub.pred_label.value_counts().to_string())
print(sub.pred_noise_type.value_counts().to_string())
