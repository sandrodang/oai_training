"""Đáp án tham khảo — BT 02: BLEU-4 kiểu SacreBLEU."""
import math
from collections import Counter


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(hyp_tokens, ref_tokens, n):
    """Trả (numerator, denominator). Numerator dùng CLIPPING:
    đếm trong hyp nhưng cắt trần bằng số lần xuất hiện trong ref."""
    hyp_ng = ngram_counts(hyp_tokens, n)
    ref_ng = ngram_counts(ref_tokens, n)
    num = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    den = sum(hyp_ng.values())
    return num, den


def brevity_penalty(c, r):
    if c == 0:
        return 0.0
    return 1.0 if c > r else math.exp(1.0 - r / c)


def corpus_bleu(hyps, refs, max_n=4, eps=1e-16, tokenize=str.split):
    """BLEU cấp corpus, thang 0-100.
    Gộp numerator/denominator TOÀN CORPUS trước khi chia — không phải trung bình BLEU từng câu.
    Smoothing: pn = (num + eps) / (den + eps), theo công thức (8) đề VOAI2025."""
    assert len(hyps) == len(refs)
    nums = [0] * (max_n + 1)
    dens = [0] * (max_n + 1)
    c_total = r_total = 0
    for hyp, ref in zip(hyps, refs):
        h = tokenize(hyp); rf = tokenize(ref)
        c_total += len(h); r_total += len(rf)
        for n in range(1, max_n + 1):
            num, den = modified_precision(h, rf, n)
            nums[n] += num; dens[n] += den
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_n = (nums[n] + eps) / (dens[n] + eps)
        log_sum += math.log(p_n)
    bp = brevity_penalty(c_total, r_total)
    return 100.0 * bp * math.exp(log_sum / max_n)
