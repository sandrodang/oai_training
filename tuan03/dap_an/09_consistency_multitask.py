"""Đáp án — R-Drop / consistency · cân bằng loss đa nhiệm."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rdrop_loss(logits1, logits2, targets, ce_fn=None, alpha=1.0):
    """R-Drop (Wu 2021): forward HAI LẦN cùng một input (dropout khác nhau),
    phạt KL đối xứng giữa hai phân phối đầu ra.

        L = CE(l1, y)/2 + CE(l2, y)/2 + alpha * KL_đối_xứng(l1, l2)/2

    Trực giác: buộc model cho cùng đáp án dù dropout tắt neuron nào -> đỡ overfit.
    Chi phí: gấp đôi forward, KHÔNG gấp đôi tham số.
    ⚠️ Phải để model ở .train() thì dropout mới khác nhau giữa hai lần.
    """
    ce_fn = ce_fn or F.cross_entropy
    ce = 0.5 * (ce_fn(logits1, targets) + ce_fn(logits2, targets))
    p, q = F.log_softmax(logits1, -1), F.log_softmax(logits2, -1)
    kl = 0.5 * (F.kl_div(p, q, log_target=True, reduction="batchmean")
                + F.kl_div(q, p, log_target=True, reduction="batchmean"))
    return ce + alpha * kl


def consistency_loss(logits_clean, logits_noisy, kind="kl"):
    """Ép dự đoán trên bản SẠCH và bản NHIỄU khớp nhau.
    Khác R-Drop ở chỗ: nhiễu đến từ AUGMENTATION dữ liệu, không phải dropout.
    Đúng thứ bài R-ViHSD cần (bản gốc vs bản teencode/bỏ dấu)."""
    if kind == "kl":
        return F.kl_div(F.log_softmax(logits_noisy, -1),
                        F.softmax(logits_clean.detach(), -1), reduction="batchmean")
    return F.mse_loss(logits_noisy, logits_clean.detach())


class UncertaintyWeighting(nn.Module):
    """Kendall & Gal (2018): HỌC trọng số cho từng nhiệm vụ thay vì chỉnh tay.

        L = Σ_i [ exp(-s_i) * L_i + s_i / 2 ]        với s_i = log(sigma_i^2)

    Nhiệm vụ nhiễu -> s_i tự tăng -> giảm trọng số. Thay cho việc bạn phải quét
    w_noise ∈ {0.15, 0.30} bằng tay như ở vòng trường.
    ⚠️ Nhớ đưa self.log_var vào optimizer — nó là THAM SỐ HỌC ĐƯỢC.
    """
    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        losses = torch.stack(list(losses))
        return (torch.exp(-self.log_var) * losses + self.log_var / 2).sum()

    def weights(self):
        return torch.exp(-self.log_var).detach()
