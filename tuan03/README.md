# TUẦN 3 · 22–28/09 · DỊCH MÁY ÍT TÀI NGUYÊN

> Thuộc `KE_HOACH_OLPAI26.md` Phần 5. Xem §4B.3 để biết tuần này nhận đơn vị kiến thức nào.

## Trạng thái

Thư mục này **mới có phần Tầng 1** (bộ đồ nghề học sâu), đã gộp vào từ `tang1_toolkit/`:

`06_weight_averaging` · `07_grad_tricks` · `08_adversarial` (FGM) · `09_consistency_multitask` (R-Drop)

Và 🔴🔴 **`10_modern_transformer`** — RMSNorm · RoPE · GQA · SwiGLU: nâng cấp Transformer
bản gốc 2017 của Tuần 2 lên bản hiện đại mà tutorial chính thức vòng Bắc 2025 dùng.
Cả bốn thành phần đều hỏng âm thầm (assert §4C **#10 · #11 · #12**).

Phần bài tập chính của tuần (`01`–`05`) sẽ được dựng khi tới tuần này.

## Chạy

```bash
cd /home/namdp36/oai/tuan03
python3 -m pytest bai_tap/test_all.py -q            # khung — đỏ hết là đúng
SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q # đáp án — phải xanh hết
```

⛔ Chỉ mở `dap_an/` sau khi đã tự thử ≥ 45 phút mỗi bài.
