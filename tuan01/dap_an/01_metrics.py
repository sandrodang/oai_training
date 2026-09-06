"""Đáp án tham khảo — BT 01: Metrics phân loại bằng numpy thuần."""
import numpy as np


def confusion_matrix_(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    # np.bincount trên chỉ số phẳng: nhanh hơn vòng lặp rất nhiều
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def precision_recall_f1(y_true, y_pred, n_classes=None):
    C = confusion_matrix_(y_true, y_pred, n_classes)
    tp = np.diag(C).astype(np.float64)
    fp = C.sum(axis=0) - tp          # cột = dự đoán
    fn = C.sum(axis=1) - tp          # hàng = thực tế
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1


def macro_f1(y_true, y_pred, n_classes=None):
    return float(precision_recall_f1(y_true, y_pred, n_classes)[2].mean())


def balanced_accuracy(y_true, y_pred, n_classes=None):
    """Trung bình recall từng lớp. Nhị phân: (TPR+TNR)/2.
    KHÔNG chứa prevalence -> nộp toàn nhãn 0 luôn cho đúng 0.5."""
    C = confusion_matrix_(y_true, y_pred, n_classes)
    tp = np.diag(C).astype(np.float64)
    support = C.sum(axis=1).astype(np.float64)
    present = support > 0                      # sklearn bỏ qua lớp không xuất hiện
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=present)
    return float(rec[present].mean())


def average_precision(y_true, scores):
    """AP theo kiểu sklearn: tổng (R_k - R_{k-1}) * P_k, KHÔNG nội suy.
    Đây là lõi của AP50 — chỉ khác ở bước ghép cặp theo IoU."""
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")   # ổn định, giảm dần
    y = y_true[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())
