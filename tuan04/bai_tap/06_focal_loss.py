"""BÀI TẬP — Hàm mất mát: Focal · Dice · Tversky · Contrastive · Triplet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    """Focal loss (Lin 2017): hạ trọng số các mẫu ĐÃ DỄ.

        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 quy về cross-entropy. gamma=2 là mặc định.
    Dùng khi mất cân bằng NẶNG. ⚠️ Với macro-F1, dịch ngưỡng sau huấn luyện
    thường rẻ hơn và mạnh ngang — nhưng nhớ bài học Tuần 1 BT05: phải kiểm bằng held-out.
    """
    raise NotImplementedError
