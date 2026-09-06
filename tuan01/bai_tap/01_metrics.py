"""BT 01 — Metrics phân loại bằng NUMPY THUẦN.  ⏱ thực tế ~60–75' · Ngày N2

Cấm import sklearn. Cấm copy. Mục tiêu là HIỂU, không phải có code chạy.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Metrics
"""
import numpy as np


def confusion_matrix_(y_true, y_pred, n_classes=None):
    """Ma trận nhầm lẫn: C[i, j] = số mẫu THỰC là i nhưng DỰ ĐOÁN là j.

    Gợi ý: đừng dùng vòng lặp. Ép cặp (i, j) thành một chỉ số phẳng
    `i * n_classes + j` rồi dùng np.bincount(..., minlength=...) và .reshape().
    """
    raise NotImplementedError


def precision_recall_f1(y_true, y_pred, n_classes=None):
    """Trả (precision, recall, f1) — mỗi cái là mảng độ dài n_classes.

    Từ ma trận nhầm lẫn C:
        tp = đường chéo
        fp = tổng theo CỘT trừ tp      (dự đoán là i nhưng thực ra không phải)
        fn = tổng theo HÀNG trừ tp     (thực là i nhưng dự đoán khác)
    Mẫu số bằng 0 -> trả 0.0 (đúng như sklearn zero_division=0), KHÔNG trả nan.
    """
    raise NotImplementedError


def macro_f1(y_true, y_pred, n_classes=None):
    """Trung bình KHÔNG TRỌNG SỐ của F1 từng lớp.

    Câu hỏi phải trả lời được: vì sao đề OlpAI luôn dùng macro chứ không dùng
    micro hay weighted?
    """
    raise NotImplementedError


def balanced_accuracy(y_true, y_pred, n_classes=None):
    """Trung bình RECALL của từng lớp. Nhị phân: (TPR + TNR) / 2.

    Lưu ý khớp sklearn: lớp KHÔNG xuất hiện trong y_true thì bỏ qua, không tính
    vào trung bình.

    Câu hỏi cốt lõi: vì sao nộp toàn nhãn 0 luôn cho ĐÚNG 0.5, bất kể tỉ lệ
    anomaly là 5% hay 50%?  (Bài học #3 trong cv/RESULTS.md của bạn.)
    """
    raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────
# average_precision / AP50 đã được HOÃN SANG TUẦN 5.
# Lý do: AP50 là metric cho detection/segmentation — xác suất ra thấp hơn
# macro-F1 và BLEU nhiều (xem KE_HOACH §3.2). Học nó lúc thật sự cần.
# Lời giải vẫn nằm sẵn ở dap_an/01_metrics.py nếu bạn tò mò.
