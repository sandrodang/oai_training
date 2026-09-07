"""BT 08 — TRẦN BAYES TỪ NHIỄU NHÃN (Tầng 0 §c2).  ⏱ ~45' · Tuần 1

Trước khi đổ giờ vào một nhiệm vụ con, hỏi: **TRẦN của nó là bao nhiêu?**
Đo trần TRƯỚC khi tối ưu, không phải sau khi đã đốt hai tiếng.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k BayesCeiling
"""
import numpy as np
from collections import Counter, defaultdict


def duplicate_groups(texts):
    """Gom chỉ số theo văn bản GIỐNG HỆT. Chỉ trả nhóm có >= 2 phần tử.
    Trả dict {text: [chỉ số...]}.
    """
    raise NotImplementedError


def ceiling_from_duplicates(texts, labels):
    """Trần accuracy suy từ mẫu TRÙNG INPUT nhưng KHÁC NHÃN.

    Với mỗi nhóm trùng input, model tốt nhất chỉ có thể đoán nhãn ĐA SỐ;
    mọi mẫu mang nhãn thiểu số trong nhóm là lỗi KHÔNG THỂ TRÁNH.

        trần = 1 - (tổng số mẫu nhãn thiểu số) / N

    Không có nhóm trùng nào -> trả 1.0.
    """
    raise NotImplementedError


def noop_rate(df, text_col="text", type_col="noise_type", original="ORIGINAL"):
    """Tỉ lệ NO-OP của từng nhãn nhiễu.

    no-op = văn bản đã "biến đổi" TRÙNG KHỚP NGUYÊN VĂN một câu ORIGINAL bất kỳ
            ⇒ phép biến đổi không để lại dấu vết nào cho model học.

    Trả dict {nhãn: tỉ lệ}. Nhãn nào tỉ lệ cao ⇒ có TRẦN CỨNG.

    🔴 NGHIỆM THU trên `work/data/training_set.csv`:
       TEENCODE **78,5%** · OBFUSCATION 9,4% · NO_DIACRITICS 7,0%
       · CHAR_REPEAT 1,7% · MIXED 1,4% · PUNCT_NOISE 0,0%
       → CHỈ TEENCODE có trần. Đó là lý do head noise ở vòng trường không thể
         cải thiện thêm, và mọi giờ đổ vào nó là lãng phí.
    """
    raise NotImplementedError
