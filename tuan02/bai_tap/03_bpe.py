"""BT 03 — BPE TỪ SỐ 0 (Sennrich et al. 2016).  ⏱ ~1h · Ngày N5

Cấm import sentencepiece / tokenizers ở bài này (dùng chúng ở phần khảo sát vocab).
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k BPE
"""
from collections import Counter, defaultdict

EOW = "</w>"          # đánh dấu hết từ — nhờ nó mà "est</w>" khác "est" giữa từ


def build_word_freq(texts):
    """Counter{từ: tần suất} trên toàn bộ texts (tách bằng khoảng trắng)."""
    raise NotImplementedError


def word_to_symbols(word):
    """"low" -> ('l','o','w','</w>').  Trả TUPLE để dùng làm khoá dict."""
    raise NotImplementedError


def get_stats(vocab):
    """vocab: {tuple(symbols): freq}  ->  Counter{(sym_a, sym_b): tổng tần suất}.

    Đếm mọi cặp LIỀN KỀ, cộng dồn theo tần suất của từ chứa nó.
    """
    raise NotImplementedError


def merge_vocab(pair, vocab):
    """Gộp mọi lần xuất hiện liền kề của `pair` thành một ký hiệu, trả vocab mới.

    ⚠️ Sau khi gộp, hai từ khác nhau có thể cho cùng một tuple -> phải CỘNG DỒN
    tần suất, đừng ghi đè.
    """
    raise NotImplementedError


def learn_bpe(texts, num_merges):
    """Trả list các cặp đã gộp, THEO ĐÚNG THỨ TỰ. Thứ tự chính là mô hình.

    Vòng lặp: get_stats -> lấy cặp tần suất cao nhất -> merge_vocab -> ghi lại.
    Dừng sớm nếu không còn cặp nào, hoặc tần suất cao nhất < 2.

    ⚠️ PHÁ HOÀ: nhiều cặp có thể cùng tần suất cao nhất. Trong ví dụ kinh điển
    {low:5, lower:2, newest:6, widest:3} có HOÀ BA CHIỀU ở 9:
    ('e','s'), ('s','t'), ('t','</w>'). Chọn cái nào là quyết định của bạn —
    không có đáp án duy nhất. Nhưng phải CỐ ĐỊNH và tái lập được (quy chế!),
    nên đừng dựa vào thứ tự duyệt dict.
    """
    raise NotImplementedError


def apply_bpe(word, merges):
    """Áp merges lên một từ, trả list ký hiệu.

    🔴 ĐIỂM DỄ SAI NHẤT: KHÔNG quét trái-sang-phải. Ở mỗi bước phải tìm cặp liền kề
    có RANK NHỎ NHẤT (học sớm nhất) trong toàn bộ chuỗi, gộp cặp đó, rồi lặp lại.

        rank = {pair: i for i, pair in enumerate(merges)}

    Ví dụ merges=[('a','b'), ('b','c')] với từ "abc":
        đúng : ['ab','c','</w>']      (rank 0 thắng rank 1)
        sai  : ['a','bc','</w>']      (nếu bạn gộp bừa cặp gặp trước)
    """
    raise NotImplementedError


def encode(text, merges):
    """Tách text theo khoảng trắng, apply_bpe từng từ, nối thành một list phẳng."""
    raise NotImplementedError


def decode(tokens):
    """Nghịch đảo của encode: nối hết rồi thay EOW bằng khoảng trắng, .strip().
    Phải KHÔNG MẤT MÁT: decode(encode(x)) == x."""
    raise NotImplementedError
