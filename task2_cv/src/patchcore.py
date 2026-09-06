"""PatchCore: coreset memory bank + kNN + image-level score. Checkpoint = memory bank."""
import numpy as np, torch

@torch.no_grad()
def greedy_coreset(feat_t, k, device, proj_dim=128, seed=1337):
    """k-center greedy tren feature giam chieu ngau nhien (Johnson-Lindenstrauss).
    Day la 'tham so hoc tu du lieu' cua mo hinh -> luu lam checkpoint."""
    n = feat_t.shape[0]
    if k >= n: return np.arange(n)
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(feat_t.shape[1], proj_dim, generator=g) / np.sqrt(proj_dim)).to(device)
    X = torch.nn.functional.normalize(feat_t.to(device).float() @ P, dim=1)
    start = int(torch.randint(n, (1,), generator=g))
    idx = [start]
    d = ((X - X[start]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(torch.argmax(d)); idx.append(i)
        d = torch.minimum(d, ((X - X[i]) ** 2).sum(1))
    return np.array(sorted(set(idx)))

@torch.no_grad()
def knn_dist(query, bank_gpu, bank_sq, k=1, chunk=16384):
    """query [Nq,D] half -> khoang cach trung binh toi k lang gieng gan nhat."""
    out = torch.empty(query.shape[0], dtype=torch.float32)
    for i in range(0, query.shape[0], chunk):
        Q = query[i:i+chunk].to(bank_gpu.device).float()
        d2 = (Q * Q).sum(1, keepdim=True) - 2 * Q @ bank_gpu.T + bank_sq[None]
        v = torch.topk(d2.clamp_(min=0), k, dim=1, largest=False).values.mean(1).sqrt()
        out[i:i+chunk] = v.cpu()
    return out

def pool_topq(d, topq=0.01):
    """d: [n_img, n_patch] -> mean cua top-q% patch cao nhat (on dinh hon .max())."""
    n_patch = d.shape[1]
    k = max(1, int(round(n_patch * topq)))
    return torch.topk(d, k, dim=1).values.mean(1)

class Bank:
    """Memory bank + kNN scoring. save()/load() = checkpoint tai lap ket qua."""
    def __init__(self, feats, device, knn_k=1):
        self.device, self.k = device, knn_k
        self.bank = feats
        self.gpu = feats.to(device).float()
        self.sq = (self.gpu * self.gpu).sum(1)
    def score_batch(self, patch, topq=0.01, keep_top=512):
        B, N, D = patch.shape
        d = knn_dist(patch.reshape(-1, D), self.gpu, self.sq, self.k).reshape(B, N)
        # giu lai top-K khoang cach patch -> cho phep quet topq/pooling OFFLINE, mien phi
        top = torch.topk(d, min(keep_top, N), dim=1).values
        return pool_topq(d, topq), top, N
    def save(self, path):
        torch.save({"bank": self.bank, "k": self.k}, path)
    @staticmethod
    def load(path, device):
        d = torch.load(path, map_location="cpu")
        return Bank(d["bank"], device, d["k"])

class Mahalanobis:
    """Nhanh phu: Gaussian tren global embedding, cov shrinkage. Tham so hoc tu train-normal."""
    def __init__(self, X, shrink=0.05):
        X = X.double()
        self.mu = X.mean(0)
        Xc = X - self.mu
        S = Xc.T @ Xc / max(1, X.shape[0] - 1)
        S += shrink * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], dtype=S.dtype)
        self.P = torch.linalg.inv(S)
    def score(self, X):
        Xc = X.double() - self.mu
        return ((Xc @ self.P) * Xc).sum(1).clamp(min=0).sqrt().float()
    def state(self): return {"mu": self.mu, "P": self.P}
