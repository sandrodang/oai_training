"""Đáp án — BT 03: BPE từ số 0 (Sennrich et al. 2016)."""
from collections import Counter, defaultdict

EOW = "</w>"


def build_word_freq(texts):
    c = Counter()
    for t in texts:
        c.update(t.split())
    return c


def word_to_symbols(word):
    return tuple(list(word) + [EOW])


def get_stats(vocab):
    """vocab: {tuple(symbols): freq} -> Counter{(a,b): freq}"""
    pairs = Counter()
    for syms, f in vocab.items():
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i + 1])] += f
    return pairs


def merge_vocab(pair, vocab):
    """Gộp mọi lần xuất hiện liền kề của `pair` trong toàn bộ vocab."""
    a, b = pair
    new = {}
    for syms, f in vocab.items():
        out, i = [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(syms[i]); i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + f
    return new


def learn_bpe(texts, num_merges):
    """Trả list các cặp đã gộp, THEO THỨ TỰ. Thứ tự chính là thuật toán."""
    wf = build_word_freq(texts)
    vocab = {word_to_symbols(w): f for w, f in wf.items()}
    merges = []
    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        best = max(stats.items(), key=lambda kv: (kv[1], kv[0]))[0]   # tie-break ổn định
        if stats[best] < 2:
            break
        merges.append(best)
        vocab = merge_vocab(best, vocab)
    return merges


def apply_bpe(word, merges):
    """Áp merges THEO ĐÚNG THỨ TỰ đã học lên một từ."""
    syms = list(word_to_symbols(word))
    rank = {p: i for i, p in enumerate(merges)}
    while len(syms) > 1:
        pairs = [(rank.get((syms[i], syms[i + 1]), None), i) for i in range(len(syms) - 1)]
        cand = [(r, i) for r, i in pairs if r is not None]
        if not cand:
            break
        _, i = min(cand)                       # luôn gộp cặp có RANK NHỎ NHẤT trước
        syms[i:i + 2] = [syms[i] + syms[i + 1]]
    return syms


def encode(text, merges):
    return [tok for w in text.split() for tok in apply_bpe(w, merges)]


def decode(tokens):
    return "".join(tokens).replace(EOW, " ").strip()
