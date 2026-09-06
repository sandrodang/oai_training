"""Đáp án tham khảo — BT 05: Tối ưu ngưỡng — khi nào giúp, khi nào hại."""
import numpy as np


def best_threshold_binary(y_true, scores, metric_fn, n_grid=None):
    """Quét mọi ngưỡng ứng viên (các giá trị score duy nhất) -> (ngưỡng, điểm).
    F1 và Balanced Accuracy KHÔNG tối ưu tại 0.5 -> luôn phải quét."""
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=np.float64)
    cands = np.unique(scores)
    if n_grid is not None and len(cands) > n_grid:
        cands = np.quantile(scores, np.linspace(0, 1, n_grid))
    # thêm hai đầu mút để cho phép dự đoán toàn 0 / toàn 1
    cands = np.concatenate([[-np.inf], cands, [np.inf]])
    best_t, best_s = 0.5, -np.inf
    for t in cands:
        s = metric_fn(y_true, (scores >= t).astype(np.int64))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


def best_class_bias(y_true, proba, metric_fn, n_iter=30, lo=-3.0, hi=3.0, n_grid=25, seed=0):
    """Tìm kiếm theo toạ độ trên vector bias cộng vào log-proba (đa lớp).
    argmax(log p + b) — dùng để hiệu chỉnh prior lệch, đúng như tune_prior.py của bạn."""
    y_true = np.asarray(y_true)
    logp = np.log(np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None))
    n_cls = logp.shape[1]
    bias = np.zeros(n_cls)
    best = metric_fn(y_true, (logp + bias).argmax(1))
    grid = np.linspace(lo, hi, n_grid)
    for _ in range(n_iter):
        improved = False
        for c in range(n_cls):
            cur = bias[c]
            for v in grid:
                bias[c] = v
                s = metric_fn(y_true, (logp + bias).argmax(1))
                if s > best + 1e-12:
                    best, cur, improved = s, v, True
            bias[c] = cur
        if not improved:
            break
    return bias, float(best)


def tuning_gain_holdout(y_true, scores, metric_fn, default_thr=0.5,
                        n_repeat=20, seed=0):
    """Giao thức half-fit / half-eval: tune ngưỡng có THẬT SỰ tổng quát hoá không?"""
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    gf, ge = [], []
    for _ in range(n_repeat):
        perm = rng.permutation(n)
        fit, ev = perm[:n // 2], perm[n // 2:]
        thr, _ = best_threshold_binary(y_true[fit], scores[fit], metric_fn)
        for idx, acc in ((fit, gf), (ev, ge)):
            tuned = metric_fn(y_true[idx], (scores[idx] >= thr).astype(np.int64))
            base = metric_fn(y_true[idx], (scores[idx] >= default_thr).astype(np.int64))
            acc.append(tuned - base)
    return float(np.mean(gf)), float(np.mean(ge))


def worth_tuning(y_true, y_pred, n_cls):
    """|precision − recall| từng lớp. Lệch lớn -> tune ngưỡng đáng làm; lệch ~0 -> bỏ qua."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    out = np.zeros(n_cls, dtype=np.float64)
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = abs(p - r)
    return out
