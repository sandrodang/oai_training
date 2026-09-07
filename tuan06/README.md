# TUẦN 6 · 13–19/10 · ML LỒNG GHÉP + TỔNG DUYỆT 1

> Thuộc `KE_HOACH_OLPAI26.md` Phần 5. Xem §4B.3 để biết tuần này nhận đơn vị kiến thức nào.

## Trạng thái

Thư mục này **mới có phần Tầng 1** (bộ đồ nghề học sâu), đã gộp vào từ `tang1_toolkit/`:

`06_finetune_lr` (LLRD · gradual unfreezing)

Phần bài tập chính của tuần (`01`–`05`) sẽ được dựng khi tới tuần này.

## Chạy

```bash
cd /home/namdp36/oai/tuan06
python3 -m pytest bai_tap/test_all.py -q            # khung — đỏ hết là đúng
SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q # đáp án — phải xanh hết
```

⛔ Chỉ mở `dap_an/` sau khi đã tự thử ≥ 45 phút mỗi bài.
