"""On-the-fly noise augmentation using the transformation rules recovered from
the organisers' own aligned pairs (see mine_rules.py).

Used for the HATE head only. Augmented rows have their noise-head loss masked
out, so the noise classifier never trains on our approximation of the
organisers' operators -- it only ever sees their real, labelled data.
"""
import re, json, random, os
from common import strip_diacritics

TOK = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
PUNCT_INSERTS = [":)))", "=)))", "@@", "!!!", "???", "..."]
OBF_MARKS = "*._"
VOWELS = "aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"


class Augmenter:
    def __init__(self, teen_rev, rng=None):
        self.teen = teen_rev                    # standard form -> list of teen forms
        self.rng = rng or random.Random(0)

    # ---- individual operators, mirroring the six labels in the task ----
    def no_diacritics(self, t):
        return strip_diacritics(t)

    def char_repeat(self, t):
        ws = t.split()
        if not ws:
            return t
        for _ in range(self.rng.randint(1, 2)):
            i = self.rng.randrange(len(ws))
            w = ws[i]
            if len(w) < 2:
                continue
            j = self.rng.randrange(len(w))
            ws[i] = w[:j] + w[j] * self.rng.randint(2, 4) + w[j + 1:]
        return " ".join(ws)

    def punct_noise(self, t):
        ws = t.split()
        blk = self.rng.choice(PUNCT_INSERTS)
        pos = self.rng.randint(0, len(ws))
        return " ".join(ws[:pos] + [" " + blk + " "] + ws[pos:]).replace("  ", "  ")

    def obfuscation(self, t):
        ws = t.split()
        cand = [i for i, w in enumerate(ws) if len(w) >= 3]
        if not cand:
            return t
        for _ in range(self.rng.randint(1, 2)):
            i = self.rng.choice(cand)
            w = ws[i]
            j = self.rng.randrange(1, len(w))
            ws[i] = w[:j] + self.rng.choice(OBF_MARKS) + w[j + 1:]
        return " ".join(ws)

    def teencode(self, t):
        out, hit = [], 0
        for w in t.split():
            lw = w.lower()
            if lw in self.teen and self.rng.random() < 0.7:
                out.append(self.rng.choice(self.teen[lw])); hit += 1
            else:
                out.append(w)
        return " ".join(out)

    def mixed(self, t):
        ops = self.rng.sample([self.no_diacritics, self.char_repeat, self.punct_noise,
                               self.obfuscation, self.teencode], self.rng.randint(2, 3))
        for op in ops:
            t = op(t)
        return t

    def __call__(self, t):
        op = self.rng.choice([self.no_diacritics, self.char_repeat, self.punct_noise,
                              self.obfuscation, self.teencode, self.mixed])
        try:
            return op(str(t))
        except Exception:
            return t


def _collapse(w):
    return re.sub(r"(.)\1+", r"\1", w)


def build(path="rules/teencode.json"):
    """Invert the mined lexicon: standard form -> the teen forms seen for it.

    The mined map was harvested from TEENCODE *and* MIXED edits, so it also
    contains char-repeat and obfuscation artefacts. Those belong to other
    operators; keeping them here would make this op produce the wrong kind of
    noise and would make `mixed` doubly noisy.
    """
    rev = {}
    if not os.path.exists(path):
        return rev
    for noisy, std in json.load(open(path)).items():
        n, d = noisy.lower(), std.lower()
        if strip_diacritics(d) == n:
            continue                                    # NO_DIACRITICS
        if any(c in n for c in OBF_MARKS):
            continue                                    # OBFUSCATION
        if _collapse(n) == d or _collapse(n) == _collapse(d):
            continue                                    # CHAR_REPEAT
        if n == d:
            continue
        rev.setdefault(d, []).append(noisy)
    return rev
