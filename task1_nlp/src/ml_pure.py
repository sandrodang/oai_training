"""Pure classical ML pipeline -- no neural network.

Evaluated on the SAME leak-free 5-fold split as the transformers, so the numbers
are directly comparable. `id` is never used as a feature.
"""
import sys, os, time, json, re, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from common import *
from features import build_clean_vocab, featurize, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="all")
ap.add_argument("--tag", default="mlpure")
a = ap.parse_args()

set_seed()
df = load_labeled(); df["fold"] = make_folds(df)
te = load_test("public_test")
TEEN = set(json.load(open("rules/teencode.json")).keys()) if os.path.exists("rules/teencode.json") else set()
TOK = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def extra_feats(texts, vocab):
    """rule features + mined-lexicon hits (inputs to a learned model, never a predictor)"""
    base = featurize(list(texts), vocab)
    hits = np.array([[sum(1 for w in TOK.findall(str(t).lower()) if w in TEEN)] for t in texts],
                    dtype=np.float32)
    return np.hstack([base, hits, np.log1p(hits)])


def build(fit_texts, fit_noise, apply_sets):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           max_features=400000, sublinear_tf=True, dtype=np.float32)
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=200000,
                           sublinear_tf=True, dtype=np.float32)
    Xc = char.fit_transform(fit_texts); Xw = word.fit_transform(fit_texts)
    vocab = build_clean_vocab([t for t, n in zip(fit_texts, fit_noise) if n == "ORIGINAL"])
    sc = StandardScaler()
    Xr = sc.fit_transform(extra_feats(fit_texts, vocab))
    X = hstack([Xc, Xw, csr_matrix(Xr)]).tocsr()
    outs = [hstack([char.transform(t), word.transform(t),
                    csr_matrix(sc.transform(extra_feats(t, vocab)))]).tocsr() for t in apply_sets]
    return X, outs, Xr, [sc.transform(extra_feats(t, vocab)) for t in apply_sets]


def zoo(ncls):
    return {
        "lr":  LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", n_jobs=-1),
        "svc": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", max_iter=3000),
                                      method="sigmoid", cv=3),
        "sgd": SGDClassifier(loss="modified_huber", alpha=1e-5, class_weight="balanced",
                             max_iter=30, random_state=SEED, n_jobs=-1),
        "nb":  ComplementNB(alpha=0.3),
    }


oof = {k: {"h": np.zeros((len(df), 3)), "n": np.zeros((len(df), 7))} for k in list(zoo(3)) + ["gbm"]}
seen = np.zeros(len(df), bool)
te_acc = {k: {"h": [], "n": []} for k in oof}
ks = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]

for k in ks:
    t0 = time.time()
    trn, val = df[df.fold != k], df[df.fold == k]
    X, (Xv, Xt), Dtr, (Dv, Dt) = build(trn.text.values, trn.noise_type.values,
                                       [val.text.values, te.text.values])
    print(f"[fold {k}] features {X.shape}  ({time.time()-t0:.0f}s)", flush=True)
    for name, mk in zoo(3).items():
        for task, y, nc in (("h", trn.y_hate.values, 3), ("n", trn.y_noise.values, 7)):
            m = zoo(nc)[name]
            Xf = X if name != "nb" else abs(X)
            m.fit(Xf, y)
            oof[name][task][val.index] = m.predict_proba(Xv if name != "nb" else abs(Xv))
            te_acc[name][task].append(m.predict_proba(Xt if name != "nb" else abs(Xt)))
        print(f"    {name} done ({time.time()-t0:.0f}s)", flush=True)
    # gradient boosting on the dense rule features only (sparse tfidf is a poor fit for trees)
    import lightgbm as lgb
    for task, y, nc in (("h", trn.y_hate.values, 3), ("n", trn.y_noise.values, 7)):
        g = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.06, num_leaves=63,
                               class_weight="balanced", random_state=SEED, n_jobs=32, verbose=-1)
        g.fit(Dtr, y)
        oof["gbm"][task][val.index] = g.predict_proba(Dv)
        te_acc["gbm"][task].append(g.predict_proba(Dt))
    seen[val.index] = True
    print(f"    gbm done ({time.time()-t0:.0f}s)", flush=True)

m = seen
print(f"\n=== per-model OOF ({m.sum()} rows) ===")
for name in oof:
    s, fh, fn = official_score(df.y_hate.values[m], oof[name]["h"][m].argmax(1),
                               df.y_noise.values[m], oof[name]["n"][m].argmax(1))
    print(f"  {name:5s} hate={fh:.4f} noise={fn:.4f} SCORE={s:.4f}")
    np.savez(f"oof/{a.tag}_{name}_oof.npz", h=oof[name]["h"], n=oof[name]["n"], seen=seen,
             te_h=np.mean(te_acc[name]["h"], 0), te_n=np.mean(te_acc[name]["n"], 0))
print(f"[saved] oof/{a.tag}_*_oof.npz")
