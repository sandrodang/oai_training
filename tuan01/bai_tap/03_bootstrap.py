"""BT 03 — Sai số chuẩn & thiên lệch chọn lọc.  ⏱ mục tiêu 50' · Ngày N3   🔴

Đây là bộ công cụ quyết định "cải thiện này có thật không" trong 6 tiếng thi.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Bootstrap
"""
import math
import numpy as np


def bootstrap_se(y_true, y_pred, metric_fn, B=1000, seed=0):
    """Sai số chuẩn của metric, ước lượng bằng bootstrap.

    Lặp B lần: lấy lại n chỉ số CÓ HOÀN LẠI -> tính metric -> lấy std(ddof=1).
    Dùng np.random.default_rng(seed) để tái lập.

    Với metric phi tuyến (macro-F1, BA, AP, BLEU) đây là cách DUY NHẤT —
    không có công thức đóng.
    """
    raise NotImplementedError


def paired_bootstrap(y_true, pred_a, pred_b, metric_fn, B=1000, seed=0):
    """So sánh hai model trên CÙNG tập test. Trả (hiệu_TB, SE_của_hiệu, P(A>B)).

    ⚠️ MẤU CHỐT: dùng CÙNG một mảng idx cho cả A và B trong mỗi vòng lặp.
    Hai dự đoán tương quan mạnh (sai ở cùng những mẫu khó), nên SE của HIỆU
    nhỏ hơn 2–3 lần so với √2 × SE biên -> phân biệt được cải thiện nhỏ mà
    SE biên tuyên bố là vô nghĩa.

    P(A>B) là đại lượng đáng dùng nhất:
        > 0.95  -> tin, đổi sang A
        0.6–0.95 -> chưa đủ, GIỮ CẢ HAI để ghép mô hình
        < 0.6   -> coi như bằng nhau, chọn cái RẺ HƠN khi inference
    """
    raise NotImplementedError


def selection_bias(se, k):
    """Thiên lệch kỳ vọng khi chọn điểm cao nhất trong k lượt nộp: ≈ SE·√(2 ln k).
    k < 2 -> 0.0.

    Thay số của chính bạn: k=20, SE=0.0116 -> +0.028,
    LỚN HƠN toàn bộ khoảng dao động giữa các cấu hình (0.702 ↔ 0.703).
    Đó là lý do KHÔNG được chọn model theo bảng public.
    """
    raise NotImplementedError


def se_scaling_check(se_small, n_small, n_big):
    """SE dự kiến ở cỡ mẫu n_big theo luật 1/√n.

    Chạy phép này MỖI LẦN bootstrap: nếu tỉ lệ SE không tuân 1/√n thì bạn
    đã cài sai (thường là resample theo FOLD thay vì theo MẪU).
    Số thật của bạn: 0.0116 @ n=3340 -> ~0.003 @ n=51663 ✓
    """
    raise NotImplementedError
