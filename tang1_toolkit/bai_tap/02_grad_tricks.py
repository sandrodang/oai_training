"""BÀI TẬP — Mẹo gradient: tích luỹ · checkpointing · thứ tự đúng với AMP."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


def accumulate_steps(target_batch, micro_batch):
    """Số bước tích luỹ để mô phỏng batch lớn trên GPU nhỏ."""
    raise NotImplementedError


def train_step_accum(model, batches, loss_fn, optimizer, scaler=None,
                     clip=1.0, scheduler=None):
    """Một bước cập nhật DUY NHẤT gộp từ nhiều micro-batch.

    🔴 HAI ĐIỂM CHẾT NGƯỜI:
      1. Chia loss cho số micro-batch. Không chia -> gradient bị nhân lên N lần,
         tương đương tăng LR N lần một cách vô tình.
      2. zero_grad MỘT LẦN trước vòng lặp, step MỘT LẦN sau vòng lặp.
         Gọi trong vòng lặp thì tích luỹ vô nghĩa.

    Thứ tự đúng khi có AMP + clipping:
        scaler.scale(loss).backward()   (mỗi micro-batch)
        scaler.unscale_(opt)            (một lần, sau vòng lặp)
        clip_grad_norm_                 (phải sau unscale, nếu không ngưỡng sai tỉ lệ)
        scaler.step(opt); scaler.update()
    """
    raise NotImplementedError


def checkpointed_forward(sequential, x, segments=2):
    """Gradient checkpointing: KHÔNG lưu activation trung gian, tính lại lúc backward.
    Đổi ~30% thời gian lấy bộ nhớ giảm mạnh — cứu cánh khi VRAM 16GB.

    ⚠️ Chỉ có tác dụng khi model đang ở chế độ train và đầu vào requires_grad,
    nếu không PyTorch cảnh báo và không lưu được gì.
    """
    raise NotImplementedError


def peak_memory_mb():
    raise NotImplementedError
