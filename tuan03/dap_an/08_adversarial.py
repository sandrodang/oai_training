"""Đáp án — Huấn luyện đối kháng: FGM · PGD · FreeLB (rút gọn).

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
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in name and p.grad is not None:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


class PGD:
    """Projected Gradient Descent — K bước nhiễu nhỏ, chiếu về quả cầu bán kính eps.
    Mạnh hơn FGM nhưng đắt gấp K lần. Trong 6 tiếng thi thường KHÔNG đáng;
    biết để hiểu FreeLB."""
    def __init__(self, model, emb_name="emb", eps=1.0, alpha=0.3):
        self.model, self.emb_name = model, emb_name
        self.eps, self.alpha = eps, alpha
        self.emb_backup, self.grad_backup = {}, {}

    def attack(self, first=False):
        for name, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in name and p.grad is not None:
                if first:
                    self.emb_backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    p.data.add_(self.alpha * p.grad / norm)
                    p.data = self._project(name, p.data)

    def _project(self, name, x):
        r = x - self.emb_backup[name]
        n = torch.norm(r)
        if n > self.eps:
            r = self.eps * r / n
        return self.emb_backup[name] + r

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.emb_backup:
                p.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        self.grad_backup = {n: p.grad.clone() for n, p in self.model.named_parameters()
                            if p.requires_grad and p.grad is not None}

    def restore_grad(self):
        for n, p in self.model.named_parameters():
            if n in self.grad_backup:
                p.grad = self.grad_backup[n]


def fgm_train_step(model, x, y, loss_fn, optimizer, emb_name="emb", eps=1.0):
    """Một bước huấn luyện đầy đủ có FGM. Trả (loss sạch, loss đối kháng)."""
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()                          # gradient SẠCH
    fgm = FGM(model, emb_name, eps)
    fgm.attack()
    loss_adv = loss_fn(model(x), y)
    loss_adv.backward()                      # CỘNG DỒN vào gradient sạch
    fgm.restore()                            # ⚠️ restore TRƯỚC step
    optimizer.step()
    return loss.item(), loss_adv.item()
