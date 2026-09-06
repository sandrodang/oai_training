"""Đáp án tham khảo — BT 03: Bootstrap SE, bootstrap ghép cặp, thiên lệch chọn lọc."""
import math
import numpy as np


def bootstrap_se(y_true, y_pred, metric_fn, B=1000, seed=0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)          # lấy lại CÓ HOÀN LẠI
        out[b] = metric_fn(y_true[idx], y_pred[idx])
    return float(out.std(ddof=1))


def paired_bootstrap(y_true, pred_a, pred_b, metric_fn, B=1000, seed=0):
    """Mấu chốt: DÙNG CÙNG idx cho cả A và B ở mỗi vòng lặp.
    Trả (hiệu trung bình, SE của hiệu, P(A tốt hơn B))."""
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)          # <-- cùng idx
        d[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    return float(d.mean()), float(d.std(ddof=1)), float((d > 0).mean())


def selection_bias(se, k):
    """Kỳ vọng thiên lệch khi chọn max trong k lượt nộp: ~ SE * sqrt(2 ln k)."""
    if k < 2:
        return 0.0
    return float(se * math.sqrt(2.0 * math.log(k)))


def se_scaling_check(se_small, n_small, n_big):
    """SE dự kiến ở cỡ mẫu n_big, theo luật 1/sqrt(n)."""
    return float(se_small * math.sqrt(n_small / n_big))
