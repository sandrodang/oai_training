"""Verify two contested EDA claims:
   (A) is OFFENSIVE really the macro-F1 bottleneck? -> need a confusion matrix
   (B) is the public-test noise excess due to MORE noisy samples,
       or merely MORE INTENSE noise per sample? (the real confound)
"""
import sys, re, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from common import *

set_seed()
df = load_labeled(); df["fold"] = make_folds(df)
pu = load_test("public_test")

# ---------------- (A) baseline confusion matrix on a single held-out group split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report

t0 = time.time()
trn, val = df[df.fold != 0], df[df.fold == 0]
vec = FeatureUnion([
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             max_features=200000, sublinear_tf=True)),
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                             sublinear_tf=True)),
])
X = vec.fit_transform(trn.text); Xv = vec.transform(val.text)
clf = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced", n_jobs=-1)
clf.fit(X, trn.y_hate)
pv = clf.predict(Xv)
print(f"[A] TF-IDF+LR baseline (fold0 held out, {time.time()-t0:.0f}s)")
print("    confusion matrix  rows=actual, cols=predicted", HATE_LABELS)
cm = confusion_matrix(val.y_hate, pv, labels=[0, 1, 2])
print(pd.DataFrame(cm, index=HATE_LABELS, columns=HATE_LABELS))
print()
print(classification_report(val.y_hate, pv, target_names=HATE_LABELS, digits=4, zero_division=0))
# where do OFFENSIVE errors go?
for i, name in enumerate(HATE_LABELS):
    row = cm[i]; tot = row.sum()
    print(f"    actual {name:9s} n={tot:5d} -> " +
          ", ".join(f"{HATE_LABELS[j]} {row[j]/tot*100:5.1f}%" for j in range(3)))
np.save("oof/baseline_cm.npy", cm)

# ---------------- (B) noise INTENSITY, conditional on the signal being present
VN = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
VN = set(VN + VN.upper())
def sig(t):
    t = str(t)
    return dict(
        rep3   = len(re.findall(r"(\w)\1{2,}", t)),
        obf    = len(re.findall(r"[A-Za-zÀ-ỹ][*._][A-Za-zÀ-ỹ]", t)),
        dspace = t.count("  "),
        diac   = sum(c in VN for c in t),
        alpha  = len(re.findall(r"[A-Za-zÀ-ỹ]", t)),
        nchar  = len(t),
    )
S  = pd.DataFrame([sig(t) for t in df.text]); S["nt"] = df.noise_type.values
Sp = pd.DataFrame([sig(t) for t in pu.text])

print("\n[B] INTENSITY given the signal fires (train-true-class vs train-all vs public)")
print(f"{'signal':<10}{'train(true class)':>20}{'train all':>12}{'public':>10}{'verdict':>26}")
for name, col, true_cls in [("rep3","rep3","CHAR_REPEAT"), ("obf","obf","OBFUSCATION"),
                            ("dspace","dspace","PUNCT_NOISE")]:
    a = S.loc[(S.nt == true_cls) & (S[col] > 0), col].mean()
    b = S.loc[S[col] > 0, col].mean()
    c = Sp.loc[Sp[col] > 0, col].mean()
    v = "same intensity" if abs(c - b) / b < 0.12 else "INTENSITY DIFFERS"
    print(f"{name:<10}{a:>20.3f}{b:>12.3f}{c:>10.3f}{v:>26}")

print("\n    length distribution (intensity-independent sanity check)")
print(f"      train chars mean={S.nchar.mean():7.2f} median={S.nchar.median():6.1f}")
print(f"      public chars mean={Sp.nchar.mean():7.2f} median={Sp.nchar.median():6.1f}")
