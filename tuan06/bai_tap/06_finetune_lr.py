"""BÀI TẬP — Fine-tune encoder pretrained: nhóm tham số · LLRD · gradual unfreezing."""
import torch
import torch.nn as nn


def param_groups_no_decay(model, weight_decay=0.01):
    """Tách 2 nhóm: KHÔNG áp weight decay lên bias và tham số norm (Bag of Tricks)."""
    raise NotImplementedError


def llrd_param_groups(layers, head, base_lr=2e-5, decay=0.95, weight_decay=0.01):
    """Layer-wise LR Decay: tầng CÀNG THẤP thì LR CÀNG NHỎ.

        lr(tầng i) = base_lr * decay^(L - 1 - i)     với L = số tầng

    Trực giác: tầng thấp học đặc trưng phổ quát (đã tốt sẵn từ pretrain), tầng cao
    học đặc trưng theo tác vụ. Ép tầng thấp học nhanh = phá kiến thức đã có
    (catastrophic forgetting). Head khởi tạo ngẫu nhiên nên để LR cao nhất.

    `layers` là list các nn.Module theo THỨ TỰ TỪ THẤP LÊN CAO.
    """
    raise NotImplementedError


def set_trainable(modules, flag):
    raise NotImplementedError


def gradual_unfreeze_schedule(n_layers, n_epochs):
    """ULMFiT: epoch 0 chỉ train head, rồi mở dần từ tầng CAO xuống THẤP.
    Trả list[int] — số tầng (tính từ trên xuống) được mở ở mỗi epoch."""
    raise NotImplementedError


def freeze_batchnorm(model):
    """Dữ liệu ít + batch nhỏ -> thống kê BatchNorm rất nhiễu. Đóng băng chúng
    (giữ running stats của pretrain) thường tốt hơn hẳn. Đặt .eval() cho lớp BN
    VÀ tắt gradient của affine params."""
    raise NotImplementedError
