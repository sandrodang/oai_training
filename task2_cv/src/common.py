"""Tien ich chung: seed, duong dan, doc test.csv."""
import os, random, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(ROOT, "dataset_train", "train")
PUBLIC_DIR = os.path.join(ROOT, "public_test")
PRIVATE_DIR= os.path.join(ROOT, "private_test")
CACHE_DIR  = os.path.join(ROOT, "cache")
SUB_DIR    = os.path.join(ROOT, "sub")
CKPT_DIR   = os.path.join(ROOT, "ckpt")
CATS = [f"category_0{i}" for i in range(1, 7)]
SEED = 1337

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def test_dir(split):
    return {"public": PUBLIC_DIR, "private": PRIVATE_DIR}[split]

def read_test_csv(split):
    """-> list of (sample_id, category, abs_path), giu nguyen thu tu file."""
    import csv
    d = test_dir(split)
    rows = []
    with open(os.path.join(d, "test.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["sample_id"], r["category"], os.path.join(d, r["relative_path"])))
    return rows

def train_files(cat):
    d = os.path.join(TRAIN_DIR, cat)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def holdout_split(cat, n_holdout=180, seed=SEED):
    """Tach train-normal thanh (fit, holdout). Holdout dung de: (1) suy ra q_c,
    (2) synthetic validation. Deterministic theo hash ten file."""
    fs = train_files(cat)
    order = sorted(range(len(fs)), key=lambda i: hashlib.md5(
        (str(seed) + os.path.basename(fs[i])).encode()).hexdigest())
    hold = set(order[:min(n_holdout, len(fs) // 3)])
    return [fs[i] for i in range(len(fs)) if i not in hold], [fs[i] for i in sorted(hold)]

for _d in (CACHE_DIR, SUB_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
