"""Sinh corpus song ngữ mô phỏng bài VOAI 2025 Tác vụ 1 (Ba Na → Việt).

Vì sao dùng ngôn ngữ nhân tạo thay vì dữ liệu thật:
  1. Bài thi là ngôn ngữ CỰC ÍT TÀI NGUYÊN, không pretrained nào giúp — đúng như đây.
  2. Có TRẦN ĐÃ BIẾT: ánh xạ là xác định, nên một Transformer đúng PHẢI đạt BLEU cao.
     Model của bạn ra BLEU thấp => code sai, không phải "bài khó". Đây là tín hiệu
     gỡ lỗi mà dữ liệu thật không cho bạn.
  3. Chạy offline, tái lập được, không phụ thuộc mạng.

Ngôn ngữ "Kơtu" (bịa) được thiết kế để KHÓ ĐÚNG CHỖ:
  - Trật tự từ SOV, trong khi tiếng Việt là SVO  -> model phải học ĐẢO thứ tự,
    không thể dịch từng từ tại chỗ.
  - Chắp dính: thì/số/phủ định là HẬU TỐ dán vào động từ/danh từ
    -> subword (BPE) thắng rõ rệt so với tách theo từ.
  - Có từ hiếm (tên riêng, số) chỉ xuất hiện 1-2 lần -> chỗ để thấy copy mechanism có ích.

    python3 make_toy_parallel.py            # sinh train/dev/test vào cùng thư mục
"""
import random, unicodedata
from pathlib import Path

SEED = 20261031
HERE = Path(__file__).parent

# ─── Từ vựng: (tiếng Việt, gốc Kơtu) ────────────────────────────────────────
NOUNS = [("con chó","ako"),("con mèo","miu"),("con gà","kata"),("con cá","hina"),
         ("cái nhà","rong"),("cái nồi","pang"),("con trâu","krao"),("cây lúa","padi"),
         ("dòng sông","dak"),("ngọn núi","bnam"),("khu rừng","brah"),("bản làng","plei"),
         ("người thầy","bok"),("đứa trẻ","kon"),("người mẹ","mi"),("người cha","pa")]
VERBS = [("ăn","cha"),("uống","nhu"),("thấy","hmang"),("bắt","gu"),("cho","ei"),
         ("làm","ngui"),("mang","tuk"),("tìm","hlao"),("nấu","om"),("bán","tec")]
ADJS  = [("to","prong"),("nhỏ","ntik"),("đẹp","liem"),("mới","mahe"),("cũ","krah")]
NAMES = [("Y Bhôk","ybhok"),("H'Linh","hling"),("Y Sơn","yson"),("H'Nga","hnga"),
         ("Y Krông","ykrong"),("H'Rin","hrin")]
NUMS  = [("một","dua"),("hai","bar"),("ba","pei"),("bốn","puon"),("năm","poh")]

# hậu tố chắp dính của Kơtu (dán liền, không cách)
PAST, PLUR, NEG = "-ne", "-tao", "-ma"


def vn_norm(s):
    return unicodedata.normalize("NFC", " ".join(s.split()))


def make_sentence(rng):
    """Trả (câu Kơtu, câu Việt)."""
    subj_is_name = rng.random() < 0.35
    if subj_is_name:
        s_vn, s_kt = rng.choice(NAMES)
    else:
        s_vn, s_kt = rng.choice(NOUNS)

    v_vn, v_kt = rng.choice(VERBS)
    o_vn, o_kt = rng.choice(NOUNS)

    past = rng.random() < 0.40
    neg  = rng.random() < 0.20
    plur = rng.random() < 0.30
    adj  = rng.random() < 0.35
    num  = rng.random() < 0.25

    # ── tân ngữ ──
    o_kt_full, o_vn_full = o_kt, o_vn
    if num:
        n_vn, n_kt = rng.choice(NUMS)
        o_kt_full = f"{o_kt} {n_kt}"          # Kơtu: DANH + SỐ
        o_vn_full = f"{n_vn} {o_vn}"          # Việt: SỐ + DANH   <- đảo thứ tự
    elif plur:
        o_kt_full = o_kt + PLUR
        o_vn_full = "những " + o_vn
    if adj:
        a_vn, a_kt = rng.choice(ADJS)
        o_kt_full = f"{o_kt_full} {a_kt}"
        o_vn_full = f"{o_vn_full} {a_vn}"

    # ── động từ: hậu tố dán liền ──
    v_kt_full = v_kt + (PAST if past else "")
    if neg:
        v_kt_full = v_kt_full + NEG

    v_vn_full = v_vn
    if past:
        v_vn_full = "đã " + v_vn_full
    if neg:
        v_vn_full = ("đã không " + v_vn) if past else ("không " + v_vn)

    kt = f"{s_kt} {o_kt_full} {v_kt_full}"     # SOV
    vn = f"{s_vn} {v_vn_full} {o_vn_full}"     # SVO
    return kt, vn_norm(vn)


def main():
    rng = random.Random(SEED)
    seen, pairs = set(), []
    while len(pairs) < 4200:                   # khử trùng lặp -> tránh rò rỉ train/test
        kt, vn = make_sentence(rng)
        if kt in seen:
            continue
        seen.add(kt); pairs.append((kt, vn))

    rng.shuffle(pairs)
    splits = {"train": pairs[:3400], "dev": pairs[3400:3800], "test": pairs[3800:]}
    for name, rows in splits.items():
        p = HERE / f"{name}.tsv"
        with p.open("w", encoding="utf-8") as f:
            f.write("src\ttgt\n")
            for kt, vn in rows:
                f.write(f"{kt}\t{vn}\n")
        print(f"{name:5s} {len(rows):5d} cặp -> {p.name}")

    ktv = {w for kt, _ in pairs for w in kt.split()}
    vnv = {w for _, vn in pairs for w in vn.split()}
    print(f"\ntừ vựng Kơtu (theo từ): {len(ktv)} | tiếng Việt: {len(vnv)}")
    print("ví dụ:")
    for kt, vn in splits["train"][:5]:
        print(f"  {kt:38s} -> {vn}")


if __name__ == "__main__":
    main()
