"""Shared utilities: data loading, grouping, scoring, seeding.

COMPLIANCE NOTES (Olympic AI 2026 rules):
  * The `id` column is NEVER used as a feature, for grouping, for embeddings,
    for rule-based prediction, or for any label inference. It is read only to
    join predictions back to test rows when writing the submission file.
  * All randomness is seeded (random / numpy / torch) for reproducibility.
  * Every model here has parameters learned from data.
"""
import os, re, random, unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = "/home/namdp36/oai/work/data"
HATE_LABELS  = ["CLEAN", "OFFENSIVE", "HATE"]
NOISE_LABELS = ["ORIGINAL", "NO_DIACRITICS", "TEENCODE", "CHAR_REPEAT",
                "PUNCT_NOISE", "OBFUSCATION", "MIXED"]
H2I = {l: i for i, l in enumerate(HATE_LABELS)}
N2I = {l: i for i, l in enumerate(NOISE_LABELS)}
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- text keys
def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def source_key(s: str) -> str:
    """Aggressively normalised form of the TEXT, used only to group the
    original/augmented pairs so that CV folds do not leak. Derived purely
    from text content -- never from `id`."""
    s = strip_diacritics(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------- loading
GROUPS_NPY = "oof/groups.npy"


def load_labeled() -> pd.DataFrame:
    """train + validation merged; deduplicated on (text,label,noise_type)."""
    tr = pd.read_csv(f"{DATA}/training_set.csv")
    va = pd.read_csv(f"{DATA}/validation_set.csv")
    tr["split"] = "train"; va["split"] = "val"
    df = pd.concat([tr, va], ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["text", "label", "noise_type"]).reset_index(drop=True)
    df["y_hate"]  = df["label"].map(H2I).astype(int)
    df["y_noise"] = df["noise_type"].map(N2I).astype(int)
    df["group"]   = df["text"].map(source_key)
    # texts that normalise to something too short get their own group
    short = df["group"].str.len() < 8
    df.loc[short, "group"] = "SHORT_" + df.index[short].astype(str)
    if os.path.exists(GROUPS_NPY):          # leak-free NN groups when available
        g = np.load(GROUPS_NPY)
        if len(g) == len(df):
            df["group"] = ["G%d" % x for x in g]
    return df


def load_test(name="public_test") -> pd.DataFrame:
    d = pd.read_csv(f"{DATA}/{name}.csv")
    d["text"] = d["text"].fillna("").astype(str)
    return d


# ---------------------------------------------------------------- folds
def make_folds(df: pd.DataFrame, n_splits=5, seed=SEED, groups=None) -> np.ndarray:
    """GroupKFold stratified on (hate x noise), grouped by source comment.

    `groups=` accepts the nearest-neighbour group ids from build_groups.py,
    which also catch OBFUSCATION variants that the string key cannot match.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    if groups is not None:
        df = df.assign(group=groups)
    strat = df["y_hate"].astype(str) + "_" + df["y_noise"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, strat, groups=df["group"])):
        fold[te] = k
    assert (fold >= 0).all()
    return fold


# ---------------------------------------------------------------- scoring
def macro_f1(y_true, y_pred, n):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n)), zero_division=0)


def official_score(yh_true, yh_pred, yn_true, yn_pred):
    fh = macro_f1(yh_true, yh_pred, 3)
    fn = macro_f1(yn_true, yn_pred, 7)
    return 0.85 * fh + 0.15 * fn, fh, fn
