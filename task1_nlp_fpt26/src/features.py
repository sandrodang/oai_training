"""Rule-derived SURFACE features.

COMPLIANCE: these are only ever *inputs* to a learned classifier, never a
predictor on their own. Every feature is computed from `text` alone -- the
`id` column is never read here.
"""
import re
import numpy as np

VN_DIAC = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
              "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")

RE_REP3    = re.compile(r"(\w)\1{2,}")
RE_REP2    = re.compile(r"([A-Za-zÀ-ỹ])\1")
RE_OBF     = re.compile(r"[A-Za-zÀ-ỹ][*._][A-Za-zÀ-ỹ]")
RE_ALPHA   = re.compile(r"[A-Za-zÀ-ỹ]")
RE_PAD     = re.compile(r"\s\s(:\)+|=\)+|@@|\.\.\.|!!!|\?\?\?|:v|:3)\s")
RE_TOK     = re.compile(r"[A-Za-zÀ-ỹ0-9]+")
RE_EMOJI   = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")

FEATURE_NAMES = [
    "len", "n_tok", "diac_ratio", "no_diac", "rep3_n", "rep3_any", "rep2_n",
    "obf_n", "obf_any", "dspace_n", "pad_n", "excl", "quest", "dots",
    "punct_ratio", "upper_ratio", "emoji_n", "digit_ratio",
    "oov_ratio", "oov_any", "shorttok_ratio",
]


def _base(t: str):
    t = str(t)
    n = max(len(t), 1)
    alpha = RE_ALPHA.findall(t)
    na = max(len(alpha), 1)
    ndiac = sum(c in VN_DIAC for c in t)
    toks = RE_TOK.findall(t.lower())
    nt = max(len(toks), 1)
    return t, n, alpha, na, ndiac, toks, nt


def build_clean_vocab(original_texts, min_count=2):
    """Lexicon learned from ORIGINAL training texts only (fit inside the fold).
    Used to measure how many tokens look 'non-standard' -> teencode signal."""
    from collections import Counter
    c = Counter()
    for t in original_texts:
        c.update(RE_TOK.findall(str(t).lower()))
    return {w for w, k in c.items() if k >= min_count}


def featurize(texts, vocab):
    out = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
    for i, raw in enumerate(texts):
        t, n, alpha, na, ndiac, toks, nt = _base(raw)
        oov = sum(1 for w in toks if w not in vocab)
        short = sum(1 for w in toks if len(w) <= 2)
        out[i] = [
            np.log1p(n), np.log1p(len(toks)), ndiac / na, float(ndiac == 0 and len(alpha) > 3),
            len(RE_REP3.findall(t)), float(bool(RE_REP3.search(t))), len(RE_REP2.findall(t)),
            len(RE_OBF.findall(t)), float(bool(RE_OBF.search(t))),
            t.count("  "), len(RE_PAD.findall(" " + t + " ")),
            t.count("!"), t.count("?"), t.count("..."),
            sum(not c.isalnum() and not c.isspace() for c in t) / n,
            sum(c.isupper() for c in t) / na,
            len(RE_EMOJI.findall(t)), sum(c.isdigit() for c in t) / n,
            oov / nt, float(oov > 0), short / nt,
        ]
    return out
