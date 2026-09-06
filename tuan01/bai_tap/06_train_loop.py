"""BT 06 — VÒNG LẶP HUẤN LUYỆN PYTORCH TỪ SỐ 0.  ⏱ N6 tra tài liệu · N8 từ trí nhớ <15'

🔴 MODULE QUAN TRỌNG NHẤT TUẦN 1. ⛔ CẤM HỎI LLM CHO FILE NÀY.

Vì sao phải tự viết thay vì dùng Lightning/HF Trainer: xem KE_HOACH §1.9.
Tóm tắt: traceback của framework dài 40+ frame, KHÔNG dán vừa cửa sổ 2.000 token
của DeepSeek trong phòng thi. Vòng lặp 120 dòng bạn tự viết thì bạn tự đọc được.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k TrainLoop
"""
# ⚠️ BẪY: GradScaler có thể BỎ QUA bước optimizer khi gradient có inf/nan (hay gặp ở
#    vài vòng đầu). Nếu bạn vẫn gọi scheduler.step() thì PyTorch sẽ cảnh báo
#    "lr_scheduler.step() before optimizer.step()" và lịch học lệch một nhịp.
#    Cách xử lý: so scaler.get_scale() trước/sau, chỉ bước scheduler khi scale không giảm.
import os, math, random, time
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────── 1. TÁI LẬP (quy chế BẮT BUỘC)
def seed_everything(seed=42, deterministic=False):
    """Đặt seed cho: random, PYTHONHASHSEED, numpy, torch, torch.cuda.
    deterministic=True -> cudnn.deterministic=True, cudnn.benchmark=False.
    Ngược lại -> benchmark=True (nhanh hơn nhưng KHÔNG xác định).
    """
    raise NotImplementedError


def seed_worker(worker_id):
    """Truyền vào DataLoader(worker_init_fn=...).
    Thiếu hàm này -> các worker dùng seed khác nhau giữa các lần chạy,
    augmentation không tái lập được -> vi phạm quy chế.
    Gợi ý: lấy torch.initial_seed() % 2**32 rồi seed numpy và random.
    """
    raise NotImplementedError


def make_loader(dataset, batch_size, shuffle, seed=42, num_workers=2):
    """DataLoader có worker_init_fn=seed_worker và generator=torch.Generator()
    đã manual_seed(seed). Bật pin_memory khi có CUDA."""
    raise NotImplementedError


# ─────────────────────────────────────────── 2. OPTIMIZER & LỊCH HỌC
def build_optimizer(model, lr=3e-4, weight_decay=0.01):
    """AdamW với 2 NHÓM tham số — 'no-bias-decay' (Bag of Tricks §4):
        nhóm 1: p.ndim > 1            -> weight_decay = weight_decay
        nhóm 2: p.ndim <= 1 hoặc bias -> weight_decay = 0.0
    (bias và weight/bias của BatchNorm/LayerNorm đều là tensor 1 chiều)

    Phải giải thích được: vì sao AdamW ≠ Adam + L2?
    """
    raise NotImplementedError


def cosine_warmup(optimizer, num_warmup, num_total, min_ratio=0.0):
    """LambdaLR: tăng tuyến tính trong num_warmup bước, rồi giảm theo cosine.
        step < num_warmup:  step / max(1, num_warmup)
        ngược lại:          min_ratio + (1-min_ratio) * 0.5 * (1 + cos(pi * progress))
    """
    raise NotImplementedError


# ─────────────────────────────────────────── 3. VÒNG LẶP
@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    """model.eval() -> trả (loss trung bình, accuracy).
    Nhớ nhân loss với len(yb) rồi chia tổng số mẫu (batch cuối có thể ngắn hơn)."""
    raise NotImplementedError


def train(model, train_loader, val_loader, *, epochs=10, lr=3e-4, weight_decay=0.01,
          warmup_ratio=0.06, patience=3, amp=True, clip=1.0, ckpt="best.pt",
          device=None, log_every=1, label_smoothing=0.0):
    """Trả (model đã nạp checkpoint tốt nhất, history).

    history là list dict, mỗi epoch một dict CÓ các khoá:
        epoch · train_loss · val_loss · val_acc · sec · mem_mb

    Phải có đủ:
      □ CrossEntropyLoss(label_smoothing=...)
      □ build_optimizer + cosine_warmup (total_steps = epochs * len(train_loader))
      □ AMP: torch.amp.GradScaler("cuda", enabled=...) + torch.amp.autocast("cuda", ...)
        ⚠️ T4 KHÔNG có bf16 -> phải fp16 + GradScaler
      □ THỨ TỰ ĐÚNG khi vừa AMP vừa clip gradient:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                      # PHẢI unscale TRƯỚC khi clip
            nn.utils.clip_grad_norm_(...)
            scaler.step(opt); scaler.update(); sched.step()
      □ model.train() đầu mỗi epoch (evaluate đã gọi .eval() -> phải bật lại!)
      □ opt.zero_grad(set_to_none=True)
      □ early stopping theo val_loss với patience
      □ lưu checkpoint tốt nhất, cuối cùng load lại
      □ đo thời gian/epoch và torch.cuda.max_memory_allocated()/2**20

    Đo thời gian và VRAM không phải để cho đẹp: tuần 1 bạn cần con số thật
    để biết T4 làm được gì trong 20 phút (ràng buộc Final/main.py).
    """
    raise NotImplementedError
