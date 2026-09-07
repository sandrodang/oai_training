"""BT 07 — VÒNG LẶP PHÂN TÍCH LỖI (Tầng 0 §c1).  ⏱ ~1h · Tuần 1

Các bài trước dạy ĐO xem cải thiện có thật không. Bài này dạy SINH RA giả thuyết
nên cải thiện CÁI GÌ. Thiếu nửa này thì mọi kỹ thuật chỉ là thử mò.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k ErrorAnalysis
"""
import numpy as np


def worst_examples(y_true, proba, k=50):
    """Chỉ số k mẫu sai NẶNG NHẤT = xác suất model gán cho lớp ĐÚNG thấp nhất.
    Trả mảng chỉ số tăng dần theo proba[i, y_true[i]] (tệ nhất đứng đầu).
    Gợi ý: np.argsort(..., kind="stable").
    """
    raise NotImplementedError


def confusion_matrix_(y_true, y_pred, n_classes=None):
    """cm[t, p] = số mẫu nhãn thật t bị đoán thành p."""
    raise NotImplementedError


def confusion_pairs(cm):
    """Mọi cặp lớp (i<j) kèm tổng nhầm lẫn và độ BẤT ĐỐI XỨNG.

    Trả list (i, j, total, asym) sắp theo total giảm dần:
        total = cm[i,j] + cm[j,i]
        asym  = |cm[i,j] - cm[j,i]| / total       (0 = đối xứng hoàn toàn)
    Bỏ qua cặp có total == 0.
    """
    raise NotImplementedError


def diagnose_pair(cm, i, j, thr=0.5):
    """🔴 Ý CHÍNH CỦA CẢ BÀI — phân biệt hai chữ ký hoàn toàn khác nhau:

      - Nhầm ĐỐI XỨNG (asym < thr): a→b nhiều VÀ b→a nhiều
        ⇒ hai lớp thật sự chồng lấn ⇒ **TRẦN**. Đừng đâm đầu tối ưu, sẽ phí giờ.
      - Nhầm MỘT CHIỀU (asym >= thr): a→b nhiều, b→a ít
        ⇒ **LỆCH PRIOR / ngưỡng sai** ⇒ sửa rất rẻ, thường chỉ vài phút.

    Trả "tran" | "lech_prior" | "khong_nham" (khi total == 0).

    ⚠️ Ở vòng trường 2026, đúng lỗi lệch prior này đã lấy mất 0,023 điểm
       (v8 được 0,697 thay vì 0,720) và nó LẶP LẠI HAI LẦN vì không ai nhìn
       ma trận nhầm lẫn.
    """
    raise NotImplementedError


def error_groups(y_true, y_pred, tags):
    """Đếm lỗi theo nhóm nguyên nhân do BẠN gán tay (tags[i] là một chuỗi).
    Trả dict {tag: số mẫu SAI mang tag đó} — chỉ đếm mẫu sai, bỏ qua mẫu đúng.

    Nhóm lớn nhất là nơi DUY NHẤT đáng đổ giờ tiếp theo.
    """
    raise NotImplementedError
