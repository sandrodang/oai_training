"""BT 08 — TRẦN BAYES TỪ NHIỄU NHÃN (Tầng 0 §c2).

Trước khi đổ giờ vào một nhiệm vụ con, hỏi: TRẦN của nó là bao nhiêu?
Đo trần TRƯỚC khi tối ưu, không phải sau.
"""
import numpy as np
from collections import Counter, defaultdict


def duplicate_groups(texts):
    """Gom chỉ số theo văn bản GIỐNG HỆT. Chỉ trả nhóm có >= 2 phần tử."""
    d = defaultdict(list)
    for i, t in enumerate(texts):
        d[t].append(i)
    return {t: idx for t, idx in d.items() if len(idx) >= 2}


def ceiling_from_duplicates(texts, labels):
    """Trần accuracy suy từ mẫu TRÙNG INPUT nhưng KHÁC NHÃN.

    Với mỗi nhóm trùng input, model tốt nhất chỉ có thể đoán nhãn ĐA SỐ.
    Mọi mẫu mang nhãn thiểu số trong nhóm là lỗi KHÔNG THỂ TRÁNH.

        trần = 1 - (tổng số mẫu nhãn thiểu số) / N
    """
    labels = list(labels)
    n = len(labels)
    lost = 0
    for idx in duplicate_groups(texts).values():
        c = Counter(labels[i] for i in idx)
        lost += len(idx) - c.most_common(1)[0][1]
    return 1.0 - lost / n if n else 1.0


def noop_rate(df, text_col="text", type_col="noise_type", original="ORIGINAL"):
    """Tỉ lệ NO-OP của từng nhãn nhiễu.

    no-op = văn bản đã "biến đổi" TRÙNG KHỚP NGUYÊN VĂN một câu ORIGINAL bất kỳ
            ⇒ phép biến đổi không để lại dấu vết nào để model học.

    Trả dict {nhãn: tỉ lệ}. Nhãn nào tỉ lệ cao ⇒ có TRẦN CỨNG, đừng tối ưu thêm.
    """
    orig = set(df.loc[df[type_col] == original, text_col])
    out = {}
    for nt, g in df[df[type_col] != original].groupby(type_col):
        out[str(nt)] = float(g[text_col].isin(orig).mean())
    return out
