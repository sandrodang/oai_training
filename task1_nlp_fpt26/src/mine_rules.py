"""Recover the organisers' transformation rules from the aligned
(ORIGINAL, noised) pairs that exist inside the TRAINING data.

Everything is derived from `text` only. The output is a lexicon + statistics
that later feed a *learned* classifier -- never used as a standalone predictor.
"""
import sys, os, re, json, difflib, unicodedata, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import *

set_seed()
df = load_labeled()

def match_form(s):
    """Form that is (approximately) invariant to ALL six transformations."""
    s = strip_diacritics(str(s)).lower()
    s = re.sub(r"[^a-z0-9]", "", s)      # kill punctuation AND obfuscation marks
    s = re.sub(r"(.)\1+", r"\1", s)      # kill char repeats
    return s

df["mf"] = df.text.map(match_form)
orig = df[df.noise_type == "ORIGINAL"].reset_index(drop=True)
nois = df[df.noise_type != "ORIGINAL"].reset_index(drop=True)
print(f"ORIGINAL={len(orig)}  noised={len(nois)}")

vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1, sublinear_tf=True)
O = normalize(vec.fit_transform(orig.mf))
N = normalize(vec.transform(nois.mf))

best_i = np.zeros(len(nois), dtype=int); best_s = np.zeros(len(nois))
CH = 2000
for a in range(0, len(nois), CH):
    S = (N[a:a+CH] @ O.T).toarray()
    best_i[a:a+CH] = S.argmax(1); best_s[a:a+CH] = S.max(1)

nois["match_idx"], nois["sim"] = best_i, best_s
print("\nnearest-ORIGINAL cosine similarity by noise type:")
print(nois.groupby("noise_type").sim.describe()[["count", "mean", "50%", "25%"]].round(3))

TH = 0.72
ok = nois[nois.sim >= TH].copy()
print(f"\naligned pairs recovered at sim>={TH}: {len(ok)} / {len(nois)}  ({len(ok)/len(nois)*100:.1f}%)")
print(ok.noise_type.value_counts().to_string())

# ------------------------------------------------------------------ word alignment
TOKR = re.compile(r"\w+|[^\w\s]", re.UNICODE)
maps = collections.defaultdict(collections.Counter)   # noise_type -> Counter[(src,dst)]
for _, r in ok.iterrows():
    src = TOKR.findall(str(orig.text.iloc[r.match_idx]).lower())
    dst = TOKR.findall(str(r.text).lower())
    sm = difflib.SequenceMatcher(a=src, b=dst, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        s_, d_ = " ".join(src[i1:i2]), " ".join(dst[j1:j2])
        if len(s_) <= 40 and len(d_) <= 40:
            maps[r.noise_type][(s_, d_)] += 1

os.makedirs("rules", exist_ok=True)
print("\n" + "=" * 74)
for nt in ["TEENCODE", "OBFUSCATION", "PUNCT_NOISE", "CHAR_REPEAT", "NO_DIACRITICS", "MIXED"]:
    c = maps[nt]
    print(f"\n### {nt}: {len(c)} distinct edits, {sum(c.values())} occurrences")
    for (s_, d_), k in c.most_common(12):
        print(f"     {k:5d}   {s_!r:28s} -> {d_!r}")

# ------------------------------------------------------------------ teencode lexicon
teen = collections.Counter()
for nt in ("TEENCODE", "MIXED"):
    for (s_, d_), k in maps[nt].items():
        if s_ and d_ and " " not in d_ and re.fullmatch(r"[\wÀ-ỹ]+", d_) and re.fullmatch(r"[\wÀ-ỹ ]+", s_):
            if d_ != s_:
                teen[(d_, s_)] += k          # noisy_form -> standard_form
lex = {}
for (d_, s_), k in teen.most_common():
    if k >= 2 and d_ not in lex:
        lex[d_] = s_
json.dump(lex, open("rules/teencode.json", "w"), ensure_ascii=False, indent=0)
print(f"\n[saved] rules/teencode.json  {len(lex)} entries (count>=2)")
print("   sample:", list(lex.items())[:25])
