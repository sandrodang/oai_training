"""Đáp án tham khảo — BT 06: Vòng lặp huấn luyện PyTorch từ số 0.

ĐÂY LÀ MODULE QUAN TRỌNG NHẤT TUẦN 1.
Mục tiêu: gõ lại được toàn bộ file này từ trí nhớ trong < 15 phút.
Không dùng Lightning / HF Trainer — xem KE_HOACH §1.9 để biết lý do.
"""
import os, math, random, time
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────── 1. TÁI LẬP (quy chế bắt buộc)
def seed_everything(seed=42, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False      # đánh đổi: chậm hơn nhưng xác định
    else:
        torch.backends.cudnn.benchmark = True       # nhanh hơn, KHÔNG xác định


def seed_worker(worker_id):
    """Truyền cho DataLoader(worker_init_fn=...) — nếu thiếu, các worker sẽ
    dùng seed khác nhau giữa các lần chạy và augmentation không tái lập được."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def make_loader(dataset, batch_size, shuffle, seed=42, num_workers=2):
    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, worker_init_fn=seed_worker, generator=g)


# ─────────────────────────────────────────────── 2. OPTIMIZER & LỊCH HỌC
def build_optimizer(model, lr=3e-4, weight_decay=0.01):
    """no-bias-decay: KHÔNG áp weight decay lên bias và tham số norm (Bag of Tricks §4)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):     # bias, BatchNorm/LayerNorm weight
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def cosine_warmup(optimizer, num_warmup, num_total, min_ratio=0.0):
    def fn(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        prog = (step - num_warmup) / max(1, num_total - num_warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ─────────────────────────────────────────────── 3. VÒNG LẶP
@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()                     # BatchNorm -> running stats, Dropout -> tắt
    tot_loss, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        tot_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return tot_loss / n, correct / n


def train(model, train_loader, val_loader, *, epochs=10, lr=3e-4, weight_decay=0.01,
          warmup_ratio=0.06, patience=3, amp=True, clip=1.0, ckpt="best.pt",
          device=None, log_every=1, label_smoothing=0.0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = build_optimizer(model, lr, weight_decay)
    total_steps = epochs * len(train_loader)
    sched = cosine_warmup(opt, int(warmup_ratio * total_steps), total_steps)
    # T4 KHÔNG có bf16 -> fp16 + GradScaler
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad, history = math.inf, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)                       # PHẢI unscale trước khi clip
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            # GradScaler có thể BỎ QUA bước optimizer khi gặp inf/nan (hay xảy ra ở
            # vài vòng đầu lúc nó dò scale). Nếu vẫn gọi sched.step() thì PyTorch cảnh
            # báo và lịch học bị lệch một nhịp -> chỉ bước scheduler khi optimizer
            # thật sự đã bước (scale không bị giảm).
            scale_truoc = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= scale_truoc:
                sched.step()
            run += loss.item() * len(yb); seen += len(yb)

        tr_loss = run / seen
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
        history.append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss,
                            val_acc=va_acc, sec=dt, mem_mb=mem))
        if log_every and ep % log_every == 0:
            print(f"ep{ep:03d} train {tr_loss:.4f} | val {va_loss:.4f} "
                  f"acc {va_acc:.4f} | {dt:.1f}s | {mem:.0f}MB | lr {sched.get_last_lr()[0]:.2e}")

        if va_loss < best - 1e-6:                          # early stopping
            best, bad = va_loss, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va_loss}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop @ep{ep} (best {best:.4f})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return model, history
