"""Denoiser built from rules mined out of the TRAINING pairs.

Produces the *normalized view* only. The raw view is always kept alongside it,
so no noise evidence is destroyed -- the encoder sees both.
Bias is deliberately toward over-normalising: hiding a toxic token (recall loss
on HATE/OFFENSIVE) costs far more than rewriting a benign one.
"""
import os, re, json, collections
from common import strip_diacritics

TOKR  = re.compile(r"[\wÀ-ỹ]+|[^\w\s]", re.UNICODE)
WORDR = re.compile(r"^[\wÀ-ỹ]+$", re.UNICODE)
PUNCT_INSERTS = [":)))", "=)))", "@@", "!!!", "???", "..."]
RE_INS    = re.compile(r"(?<!\S)(" + "|".join(re.escape(p) for p in PUNCT_INSERTS) + r")(?!\S)")
RE_DIAC   = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
OBF_CHARS = "*._"
VN_LETTERS = "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệghiìíỉĩịklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvxyỳýỷỹỵ"


class Normalizer:
    def __init__(self, vocab, teen, nodiac):
        self.vocab, self.teen, self.nodiac = vocab, teen, nodiac

    def _word(self, w, force_diac=False):
        lw = w.lower()
        if lw in self.teen:                                   # mined teencode wins first
            return self.teen[lw]
        if force_diac and lw in self.nodiac:                  # whole text lost its accents
            return self.nodiac[lw]
        if lw in self.vocab:
            return w
        c = re.sub(r"(.)\1{1,}", r"\1", lw)                   # undo char repeat
        if c in self.vocab:
            return c
        if re.sub(r"(.)\1{2,}", r"\1\1", lw) in self.vocab:
            return re.sub(r"(.)\1{2,}", r"\1\1", lw)
        if any(ch in lw for ch in OBF_CHARS):                 # undo obfuscation
            base = "".join(ch for ch in lw if ch not in OBF_CHARS)
            if base in self.vocab:
                return base
            for i, ch in enumerate(lw):
                if ch in OBF_CHARS:
                    for r in VN_LETTERS:
                        cand = lw[:i] + r + lw[i + 1:]
                        if cand in self.vocab:
                            return cand
            if base in self.nodiac:
                return self.nodiac[base]
        if lw in self.nodiac:                                 # restore diacritics
            return self.nodiac[lw]
        if c in self.nodiac:
            return self.nodiac[c]
        return w

    def __call__(self, text):
        t = str(text)
        t = RE_INS.sub(" ", t)                                # drop the 6 inserted blocks
        t = re.sub(r"\s+", " ", t).strip()
        force = RE_DIAC.search(t) is None                     # text carries no accents at all
        toks = TOKR.findall(t)
        return " ".join(self._word(x, force) if WORDR.match(x) else x for x in toks)


def build(df, min_count=2):
    cnt = collections.Counter()
    for t in df.loc[df.noise_type == "ORIGINAL", "text"]:
        cnt.update(w.lower() for w in TOKR.findall(str(t)) if WORDR.match(w))
    vocab = {w for w, k in cnt.items() if k >= min_count}

    by_strip = collections.defaultdict(collections.Counter)
    for w, k in cnt.items():
        if k >= min_count:
            by_strip[strip_diacritics(w).lower()][w] += k
    nodiac = {}
    for s, c in by_strip.items():
        top, n = c.most_common(1)[0]
        if top != s and n / sum(c.values()) >= 0.55:
            nodiac[s] = top

    teen = {}
    if os.path.exists("rules/teencode.json"):
        for noisy, std in json.load(open("rules/teencode.json")).items():
            # drop pure diacritic-stripping edits: they belong to NO_DIACRITICS
            # and keeping them here would blur the two noise classes
            if strip_diacritics(std).lower() == noisy.lower():
                continue
            teen[noisy] = std
    return Normalizer(vocab, teen, nodiac), dict(vocab=len(vocab), teen=len(teen), nodiac=len(nodiac))
