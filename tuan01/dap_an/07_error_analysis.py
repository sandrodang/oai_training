"""BT 07 — VÒNG LẶP PHÂN TÍCH LỖI (Tầng 0 §c1).

Tầng 0 các bài trước dạy ĐO xem cải thiện có thật không.
Bài này dạy SINH RA giả thuyết nên cải thiện cái gì. Thiếu nửa này thì
mọi kỹ thuật chỉ là thử mò theo thứ tự ngẫu nhiên.
"""
import numpy as np


def worst_examples(y_true, proba, k=50):
    """Chỉ số k mẫu sai NẶNG NHẤT: xác suất model gán cho lớp ĐÚNG thấp nhất.

    Trả mảng chỉ số, tăng dần theo proba[i, y_true[i]] (tệ nhất đứng đầu).
    """
    y_true = np.asarray(y_true)
    p_true = np.asarray(proba)[np.arange(len(y_true)), y_true]
    return np.argsort(p_true, kind="stable")[:k]


def confusion_matrix_(y_true, y_pred, n_classes=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def confusion_pairs(cm):
    """Mọi cặp lớp (i<j) kèm tổng nhầm lẫn và độ BẤT ĐỐI XỨNG.

    Trả list (i, j, total, asym) sắp theo total giảm dần, với
        total = cm[i,j] + cm[j,i]
        asym  = |cm[i,j] - cm[j,i]| / total      (0 = đối xứng hoàn toàn)
    """
    cm = np.asarray(cm)
    out = []
    for i in range(len(cm)):
        for j in range(i + 1, len(cm)):
            tot = int(cm[i, j] + cm[j, i])
            if tot == 0:
                continue
            out.append((i, j, tot, abs(int(cm[i, j]) - int(cm[j, i])) / tot))
    return sorted(out, key=lambda r: -r[2])


def diagnose_pair(cm, i, j, thr=0.5):
    """Phân loại một cặp nhầm lẫn thành 'tran' hay 'lech_prior'.

    🔴 ĐÂY LÀ Ý CHÍNH CỦA CẢ BÀI:
      - Nhầm ĐỐI XỨNG (asym < thr): a→b nhiều VÀ b→a nhiều
        ⇒ hai lớp thật sự chồng lấn ⇒ **TRẦN**, đừng đâm đầu tối ưu.
      - Nhầm MỘT CHIỀU (asym >= thr): a→b nhiều, b→a ít
        ⇒ **LỆCH PRIOR / ngưỡng sai** ⇒ sửa rất rẻ, thường vài phút.
    """
    cm = np.asarray(cm)
    tot = int(cm[i, j] + cm[j, i])
    if tot == 0:
        return "khong_nham"
    asym = abs(int(cm[i, j]) - int(cm[j, i])) / tot
    return "lech_prior" if asym >= thr else "tran"


def error_groups(y_true, y_pred, tags):
    """Đếm lỗi theo nhóm nguyên nhân do BẠN gán nhãn tay (tags[i] là một chuỗi).

    Trả dict {tag: số mẫu SAI mang tag đó}, để biết nhóm nào lớn nhất —
    nhóm lớn nhất là nơi DUY NHẤT đáng đổ giờ tiếp theo.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for i in np.nonzero(y_true != y_pred)[0]:
        out[tags[i]] = out.get(tags[i], 0) + 1
    return out
