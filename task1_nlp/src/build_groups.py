"""Assign every row to its SOURCE COMMENT group via nearest-neighbour matching.

The exact-string key in common.py cannot group OBFUSCATION variants (one
character is replaced, so no normalisation recovers the original spelling).
Those rows therefore leaked across CV folds. This rebuilds the groups with a
char-ngram nearest-neighbour search against the ORIGINAL rows.
Derived from `text` only.
"""
import sys, os, re; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import *

set_seed()
df = load_labeled()

def mf(s):
    s = strip_diacritics(str(s)).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)

df["mf"] = df.text.map(mf)
is_o = (df.noise_type == "ORIGINAL").values
oi = np.where(is_o)[0]; ni = np.where(~is_o)[0]

vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
O = normalize(vec.fit_transform(df.mf.values[oi]))
N = normalize(vec.transform(df.mf.values[ni]))

grp = np.arange(len(df))
# merge ORIGINAL duplicates first
first = {}
for j, i in enumerate(oi):
    k = df.mf.iat[i]
    grp[i] = first.setdefault(k, i)

TH = 0.70
best_i = np.zeros(len(ni), int); best_s = np.zeros(len(ni))
for a in range(0, len(ni), 2000):
    S = (N[a:a+2000] @ O.T).toarray()
    best_i[a:a+2000] = S.argmax(1); best_s[a:a+2000] = S.max(1)

hit = best_s >= TH
grp[ni[hit]] = grp[oi[best_i[hit]]]
print(f"noised rows matched to an ORIGINAL: {hit.sum()}/{len(ni)} ({hit.mean()*100:.1f}%)")
print("match rate by noise type:")
print(pd.Series(hit, index=df.noise_type.values[ni]).groupby(level=0).mean().round(3).to_string())

old = pd.Series(df.text.map(source_key).values)
old = old.where(old.str.len() >= 8, pd.Series(["S%d" % i for i in range(len(df))]))
print(f"\ngroups: old key = {old.nunique()}   new NN = {pd.Series(grp).nunique()}")
np.save("oof/groups.npy", grp)
print("[saved] oof/groups.npy")
