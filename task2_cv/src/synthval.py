"""Bo validation CO NHAN THAT tu chinh du lieu train (de bai cho phep ro).

Tao anomaly tong hop tren anh train-normal held-out, roi do AUROC THAT.
Khac han AUC_proxy: khong bi nhiem boi chenh lech mien nen hay ti le anomaly.
Dung lam ham muc tieu cho random search.
"""
import os, sys, argparse, io
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS, CACHE_DIR, CKPT_DIR, ROOT, set_seed, holdout_split
from features import iter_feats
from patchcore import Bank
from proxy import auc

def make_anomaly(im, rng, kind):
    """5 kieu hong mo phong cac dang loi thuong gap trong kiem tra ngoai quan."""
    w, h = im.size; im = im.copy()
    s = int(rng.integers(int(0.03 * min(w, h)), int(0.12 * min(w, h))))
    x, y = int(rng.integers(0, w - s)), int(rng.integers(0, h - s))
    if kind == "cutpaste":                       # dan mieng khac cua chinh anh
        x2, y2 = int(rng.integers(0, w - s)), int(rng.integers(0, h - s))
        im.paste(im.crop((x2, y2, x2 + s, y2 + s)), (x, y))
    elif kind == "scar":                         # vet manh dai
        d = ImageDraw.Draw(im)
        d.line([(x, y), (x + int(rng.integers(-3*s, 3*s)), y + int(rng.integers(-3*s, 3*s)))],
               fill=tuple(int(v) for v in rng.integers(0, 255, 3)), width=int(rng.integers(2, 6)))
    elif kind == "blur":                         # mat net cuc bo
        r = im.crop((x, y, x + s, y + s)).filter(ImageFilter.GaussianBlur(float(s) / 5.0))
        im.paste(r, (x, y))
    elif kind == "color":                        # lech mau cuc bo
        r = np.asarray(im.crop((x, y, x + s, y + s))).astype(np.int16)
        r = np.clip(r + rng.integers(-70, 70, 3)[None, None], 0, 255).astype(np.uint8)
        im.paste(Image.fromarray(r), (x, y))
    elif kind == "erase":                        # thieu chi tiet: to bang mau nen
        bg = np.asarray(im).reshape(-1, 3)
        med = tuple(int(v) for v in np.median(bg, 0))
        ImageDraw.Draw(im).rectangle([x, y, x + s, y + s], fill=med)
    return im

KINDS = ["cutpaste", "scar", "blur", "color", "erase"]

def build_set(files, out_dir, seed=1337):
    """Sinh 1 anh anomaly cho moi anh holdout, luan phien kieu hong."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        a = make_anomaly(im, rng, KINDS[i % len(KINDS)])
        p = os.path.join(out_dir, f"anom_{i:04d}.jpg")
        a.save(p, "JPEG", quality=95)
        paths.append(p)
    return paths

@torch.no_grad()
def eval_bank(tag, model, long_side, dev, topq=0.01, bs=4):
    """AUROC THAT: holdout-normal vs synthetic-anomaly."""
    per = {}
    for cat in CATS:
        _, hold_f = holdout_split(cat)
        adir = os.path.join(CACHE_DIR, "synth", cat)
        anom_f = (sorted(os.path.join(adir, x) for x in os.listdir(adir))
                  if os.path.isdir(adir) and os.listdir(adir) else build_set(hold_f, adir))
        bank = Bank.load(os.path.join(CKPT_DIR, tag, f"{cat}_bank.pt"), dev)
        sc = {}
        for nm, fs in (("n", hold_f), ("a", anom_f)):
            out = []
            for patch, _ in iter_feats(fs, model, long_side, dev, None, bs):
                out.append(bank.score_batch(patch, topq)[0])
            sc[nm] = torch.cat(out).numpy()
        per[cat] = auc(sc["a"], sc["n"])
        del bank; torch.cuda.empty_cache()
        print(f"  {cat}: AUROC_that = {per[cat]:.4f}", flush=True)
    return per

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--long", type=int, required=True); ap.add_argument("--topq", type=float, default=0.01)
    a = ap.parse_args(); set_seed()
    per = eval_bank(a.tag, a.model, a.long, torch.device("cuda"), a.topq)
    print(f"\n  TRUNG BINH AUROC that = {np.mean(list(per.values())):.4f}")
