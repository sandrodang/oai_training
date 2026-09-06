# 🥊 DRILL — GÕ LẠI TỪ TRÍ NHỚ
### ⏸ ĐÃ HOÃN — chưa làm ở Tuần 1

> **Vì sao hoãn:** bài này tốn 3 giờ, và nó chỉ CẦN THIẾT nếu **không được mang code
> vào phòng thi**. Câu hỏi đó đang chờ BTC trả lời (`KE_HOACH_OLPAI26.md` §11.4).
>
> - BTC trả lời **KHÔNG được mang code** → drill thành **bắt buộc**, làm ở Tuần 6 và Tuần 8.
> - BTC trả lời **được mang** → drill hạ xuống tuỳ chọn; thay bằng việc chuẩn bị
>   một **thư viện tiện ích sạch** để mang theo.
>
> Dù thế nào, `train_loop.py` vẫn phải **tự viết ở N7 Tuần 1** — hiểu nó là bắt buộc,
> chỉ có việc *thuộc lòng* mới hoãn.

> **Vì sao có bài này:** trong phòng thi bạn chỉ có DeepSeek **2.000 token ngữ cảnh**.
> Không dán nổi một traceback framework. Thứ bạn *thuộc* mới là thứ bạn *dùng được*.
> Đây không phải khổ hạnh — đây là điều kiện để sống sót 6 tiếng.

---

## Luật

1. **Màn hình trắng.** Đóng hết `dap_an/`, `bai_tap/`, tài liệu, trình duyệt.
2. **Cấm LLM tuyệt đối.** Cấm cả tra `pytorch.org`.
3. **Bấm giờ từ khi gõ ký tự đầu tiên** đến khi `pytest` xanh.
4. Được xem **danh sách hàm cần viết** dưới đây — đó là tất cả.
5. Sai thì tự sửa. Bí quá **> 10 phút** ở một chỗ: mở đáp án, đọc **đúng hàm đó**,
   đóng lại, **xoá hết và gõ lại từ đầu**. Ghi vào bảng là lượt đó "có mở đáp án".

---

## Vòng 1 — `seed_everything` + `build_optimizer` + `cosine_warmup` · đích **5 phút**

```
seed_everything(seed, deterministic)   # random, PYTHONHASHSEED, numpy, torch, cuda, cudnn
seed_worker(worker_id)
make_loader(dataset, bs, shuffle, seed, num_workers)
build_optimizer(model, lr, weight_decay)    # 2 nhóm: no-bias-decay
cosine_warmup(optimizer, num_warmup, num_total, min_ratio)
```

Kiểm: `python3 -m pytest bai_tap/test_all.py -q -k "seed or optimizer or scheduler"`

## Vòng 2 — `evaluate` + `train` đầy đủ · đích **15 phút** *(tính cả vòng 1)*

```
evaluate(model, loader, loss_fn, device)
train(model, tr, va, epochs, lr, weight_decay, warmup_ratio,
      patience, amp, clip, ckpt, device, log_every, label_smoothing)
```

**7 điểm dễ quên — tự kiểm trước khi chạy:**
- [ ] `model.train()` ở ĐẦU mỗi epoch (vì `evaluate` đã gọi `.eval()`)
- [ ] `opt.zero_grad(set_to_none=True)`
- [ ] `scaler.unscale_(opt)` **TRƯỚC** `clip_grad_norm_`
- [ ] `sched.step()` **sau** `scaler.step(opt)`
- [ ] loss nhân `len(yb)` rồi chia tổng số mẫu (batch cuối ngắn hơn)
- [ ] `total_steps = epochs * len(train_loader)`, không phải `epochs`
- [ ] cuối cùng `load_state_dict` checkpoint tốt nhất

Kiểm: `python3 -m pytest bai_tap/test_all.py -q -k TrainLoop`

## Vòng 3 — Metrics · đích **8 phút**

```
confusion_matrix_ · precision_recall_f1 · macro_f1 · balanced_accuracy · average_precision
brevity_penalty · modified_precision · corpus_bleu
bootstrap_se · paired_bootstrap · selection_bias
```

Kiểm: `python3 -m pytest bai_tap/test_all.py -q -k "Metrics or BLEU or Bootstrap"`

---

## Chuẩn đạt

| Vòng | Đích | Đạt nếu |
|---|---|---|
| 1 | 5' | xanh, không mở đáp án |
| 2 | 15' | xanh, không mở đáp án, **chạy đúng ngay lần đầu** |
| 3 | 8' | xanh, không mở đáp án |

**Quá đích 2 lần liên tiếp = chưa thuộc** → lặp lại vào tuần 6 và tuần 8 (kế hoạch đã xếp sẵn lịch ôn 3 lần).
