"""BT 02 — BLEU-4 kiểu SacreBLEU.  ⏱ mục tiêu 75' · Ngày N2   🔴 ƯU TIÊN CAO NHẤT

SacreBLEU xuất hiện ở 2/2 đề mẫu 2025; VOAI CK cho trọng số 0.8.
Cấm import sacrebleu/nltk.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k BLEU
"""
import math
from collections import Counter


def ngram_counts(tokens, n):
    """Counter các n-gram (dạng tuple) trong danh sách token.
    Số n-gram của chuỗi dài L là L - n + 1 (âm thì coi như rỗng)."""
    raise NotImplementedError


def modified_precision(hyp_tokens, ref_tokens, n):
    """Trả (numerator, denominator) cho n-gram bậc n.

    CLIPPING là ý tưởng cốt lõi của BLEU:
        numerator   = Σ_g min( count_hyp(g), count_ref(g) )
        denominator = Σ_g count_hyp(g)
    Nhờ min(...) mà lặp một từ đúng 10 lần không ăn thêm điểm.
    """
    raise NotImplementedError


def brevity_penalty(c, r):
    """c = tổng độ dài bản dịch, r = tổng độ dài tham chiếu.
        BP = 1                  nếu c > r
        BP = exp(1 − r/c)       nếu c ≤ r
    c == 0 -> trả 0.0.

    Hiểu cho kỹ: phạt theo hàm MŨ. Model thiếu dữ liệu hay dịch ngắn cho 'an toàn'
    và BLEU trừng phạt rất nặng hành vi đó.
    """
    raise NotImplementedError


def corpus_bleu(hyps, refs, max_n=4, eps=1e-16, tokenize=str.split):
    """BLEU cấp corpus, thang 0–100.

    ⚠️ BẪY: phải GỘP numerator/denominator TOÀN CORPUS rồi mới chia —
    KHÔNG phải trung bình BLEU của từng câu. Hai cách cho kết quả khác nhau.

        p_n  = (Σ num_n + eps) / (Σ den_n + eps)        # smoothing, công thức (8) đề VOAI2025
        BLEU = 100 · BP · exp( (1/N) Σ_{n=1..N} log p_n )

    BLEU là trung bình NHÂN -> một p_n = 0 kéo cả điểm về 0, nên cần eps.
    """
    raise NotImplementedError
