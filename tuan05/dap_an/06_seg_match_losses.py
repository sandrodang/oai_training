"""Đáp án — Hàm mất mát: Focal · Dice · Tversky · Contrastive · Triplet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1.0):
    """Dice loss cho segmentation nhị phân. logits & targets: (B, ...) cùng shape.

        Dice = 2|X∩Y| / (|X| + |Y|)      ->  loss = 1 - Dice

    Vì sao dùng cho segmentation: BCE bị thống trị bởi nền (95% pixel là nền),
    Dice đo TRÙNG KHỚP vùng nên không bị lệch bởi mất cân bằng lớp.
    Thực tế hầu như luôn dùng **BCE + Dice** cộng lại.
    """
    p = torch.sigmoid(logits).flatten(1)
    t = targets.float().flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=1.0):
    """Tổng quát hoá Dice, chỉnh cán cân FP/FN.
        alpha phạt FP · beta phạt FN.  alpha=beta=0.5 -> Dice.
    beta > alpha  ->  THIÊN VỀ RECALL (bỏ sót tổn thương đắt hơn báo nhầm).
    Đúng thứ cần cho bài phát hiện vùng bệnh chấm bằng AP50.
    """
    p = torch.sigmoid(logits).flatten(1)
    t = targets.float().flatten(1)
    tp = (p * t).sum(1); fp = (p * (1 - t)).sum(1); fn = ((1 - p) * t).sum(1)
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def contrastive_loss(z1, z2, label, margin=1.0):
    """Siamese: label=1 nghĩa là CÙNG cặp (kéo lại gần), 0 là khác (đẩy ra xa margin).
        L = y*d^2 + (1-y)*max(0, margin - d)^2
    Dùng cho bài ghép ảnh: học "hai mảnh này có kề nhau không" (VOAI CK TV2).
    """
    d = F.pairwise_distance(z1, z2)
    return (label * d.pow(2) + (1 - label) * F.relu(margin - d).pow(2)).mean()


def triplet_loss(anchor, positive, negative, margin=1.0):
    """L = max(0, d(a,p) - d(a,n) + margin). Không cần nhãn nhị phân, chỉ cần bộ ba."""
    return F.relu(F.pairwise_distance(anchor, positive)
                  - F.pairwise_distance(anchor, negative) + margin).mean()
