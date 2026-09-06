"""BÀI TẬP — Huấn luyện đối kháng: FGM · PGD · FreeLB (rút gọn).

Ý tưởng chung: thêm nhiễu NHỎ theo hướng LÀM TĂNG loss vào embedding, rồi bắt
model vẫn dự đoán đúng. Kết quả: biên quyết định phẳng hơn, chịu nhiễu tốt hơn.

Đúng thứ đề R-ViHSD cần (teencode, bỏ dấu, che ký tự). Bạn đã dùng FGM ở vòng
trường: fold0 0.7114 -> 0.7277, tức +0.016.
"""
import torch


class FGM:
    """Fast Gradient Method — RẺ NHẤT: chỉ thêm 1 lần forward/backward.

        r = eps * g / ||g||        (g = gradient của embedding)

    Quy trình mỗi batch:
        loss.backward()          # gradient sạch, tích luỹ
        fgm.attack()             # cộng nhiễu vào embedding
        loss_adv.backward()      # gradient đối kháng, CỘNG DỒN vào gradient sạch
        fgm.restore()            # trả embedding về cũ
        optimizer.step()
    """
    def __init__(self, model, emb_name="emb", eps=1.0):
        raise NotImplementedError

    def attack(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError


class PGD:
    """Projected Gradient Descent — K bước nhiễu nhỏ, chiếu về quả cầu bán kính eps.
    Mạnh hơn FGM nhưng đắt gấp K lần. Trong 6 tiếng thi thường KHÔNG đáng;
    biết để hiểu FreeLB."""
    def __init__(self, model, emb_name="emb", eps=1.0, alpha=0.3):
        raise NotImplementedError

    def attack(self, first=False):
        raise NotImplementedError

    def _project(self, name, x):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError

    def backup_grad(self):
        raise NotImplementedError

    def restore_grad(self):
        raise NotImplementedError


def fgm_train_step(model, x, y, loss_fn, optimizer, emb_name="emb", eps=1.0):
    """Một bước huấn luyện đầy đủ có FGM. Trả (loss sạch, loss đối kháng)."""
    raise NotImplementedError
