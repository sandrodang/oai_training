"""Baseline submission: TF-IDF (char+word) + rule features -> two Logistic
Regressions (hate 3-way, noise 7-way).

Real ML model (learned coefficients), fully seeded, `id` used only to join
predictions back to the test rows when writing the CSV.
"""
import sys, os, time, argparse, zipfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from features import build_clean_vocab, featurize

ap = argparse.ArgumentParser()
ap.add_argument("--test", default="public_test")
ap.add_argument("--phase", default="public", choices=["public", "private"])
ap.add_argument("--outdir", default="sub")
a = ap.parse_args()

set_seed(SEED)
t0 = time.time()


def build_matrices(fit_texts, fit_noise, apply_sets):
    """Fit vectorisers + lexicon on fit_texts only, then transform each set."""
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_features=300000, sublinear_tf=True)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000,
                           sublinear_tf=True)
    Xc = char.fit_transform(fit_texts); Xw = word.fit_transform(fit_texts)
    vocab = build_clean_vocab([t for t, n in zip(fit_texts, fit_noise) if n == "ORIGINAL"])
    sc = StandardScaler()
    Xr = sc.fit_transform(featurize(list(fit_texts), vocab))
    X = hstack([Xc, Xw, csr_matrix(Xr)]).tocsr()
    outs = []
    for ts in apply_sets:
        o = hstack([char.transform(ts), word.transform(ts),
                    csr_matrix(sc.transform(featurize(list(ts), vocab)))]).tocsr()
        outs.append(o)
    return X, outs


def fit_predict(X, yh, yn, Xs):
    mh = LogisticRegression(max_iter=1500, C=4.0, class_weight="balanced", n_jobs=-1)
    mn = LogisticRegression(max_iter=1500, C=4.0, class_weight="balanced", n_jobs=-1)
    mh.fit(X, yh); mn.fit(X, yn)
    return [(mh.predict_proba(x), mn.predict_proba(x)) for x in Xs]


# ---------------------------------------------------------------- 1. honest eval
df = load_labeled(); df["fold"] = make_folds(df)
trn, val = df[df.fold != 0], df[df.fold == 0]
X, (Xv,) = build_matrices(trn.text.values, trn.noise_type.values, [val.text.values])
(ph, pn), = fit_predict(X, trn.y_hate.values, trn.y_noise.values, [Xv])
sc_, fh, fn = official_score(val.y_hate.values, ph.argmax(1),
                             val.y_noise.values, pn.argmax(1))
print(f"[eval] held-out group split (fold 0, n={len(val)})")
print(f"       hate  macro-F1 = {fh:.4f}")
print(f"       noise macro-F1 = {fn:.4f}")
print(f"       OFFICIAL SCORE = {sc_:.4f}   (0.85*hate + 0.15*noise)")
np.save("oof/baseline_fold0_hate.npy", ph); np.save("oof/baseline_fold0_noise.npy", pn)

# ---------------------------------------------------------------- 2. full refit
te = load_test(a.test)
Xa, (Xt,) = build_matrices(df.text.values, df.noise_type.values, [te.text.values])
(ph_t, pn_t), = fit_predict(Xa, df.y_hate.values, df.y_noise.values, [Xt])

sub = pd.DataFrame({
    "id": te["id"],                                   # join key only, not a feature
    "pred_label":      [HATE_LABELS[i]  for i in ph_t.argmax(1)],
    "pred_noise_type": [NOISE_LABELS[i] for i in pn_t.argmax(1)],
})
os.makedirs(a.outdir, exist_ok=True)
csv_name = f"task1_{a.phase}_output.csv"
csv_path = os.path.join(a.outdir, csv_name)
sub.to_csv(csv_path, index=False, encoding="utf-8")
np.save(f"oof/baseline_{a.phase}_hate.npy", ph_t); np.save(f"oof/baseline_{a.phase}_noise.npy", pn_t)

print(f"\n[write] {csv_path}  rows={len(sub)}")
print("        pred_label distribution:");      print(sub.pred_label.value_counts().to_string())
print("        pred_noise_type distribution:"); print(sub.pred_noise_type.value_counts().to_string())
print(f"\n[done] {time.time()-t0:.0f}s")
