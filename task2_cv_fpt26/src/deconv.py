"""Nguong rieng tung category, KHONG ton luot nop.

test = (1-pi)*normal + pi*anomaly, va ta BIET phan bo normal (holdout train-normal).
Voi moi nguong t:
    TNR(t) = P(normal < t)                              <- do truc tiep tu holdout
    TPR(t) = [S_test(t) - (1-pi) S_norm(t)] / pi        <- suy ra tu dang thuc hon hop
=> BA(t) tinh duoc cho moi t ma khong can nhan nao.

pi_c uoc luong tu rang buoc TPR(t) <= 1:
    pi >= [S_test(t) - S_norm(t)] / [1 - S_norm(t)]  voi moi t
dung luong tu hoa robust thay vi max tuyet doi de tranh nhieu duoi.
"""
import numpy as np

def surv(x, t):
    return (x[None, :] >= t[:, None]).mean(1)

def estimate_pi(te, ho, grid, robust_q=0.95, lo=0.05, hi=0.995):
    St, Sn = surv(te, grid), surv(ho, grid)
    m = (Sn < hi) & (Sn > 1 - hi) | True
    r = (St - Sn) / np.maximum(1 - Sn, 1e-6)
    # chi xet vung nguong co y nghia (khong qua cuc doan hai dau)
    ok = (St > lo) & (St < 1 - lo / 5) & np.isfinite(r)
    if ok.sum() < 5: ok = np.isfinite(r)
    return float(np.clip(np.quantile(r[ok], robust_q), 0.02, 0.95))

def ba_curve(te, ho, grid, pi):
    St, Sn = surv(te, grid), surv(ho, grid)
    tpr = np.clip((St - (1 - pi) * Sn) / max(pi, 1e-6), 0, 1)
    tnr = 1 - Sn
    return 0.5 * (tpr + tnr), tpr, tnr

def best_tau(te, ho, pi=None, ngrid=400, robust_q=0.95, smooth=9):
    lo, hi = min(te.min(), ho.min()), max(te.max(), ho.max())
    grid = np.linspace(lo, hi, ngrid)
    if pi is None: pi = estimate_pi(te, ho, grid, robust_q)
    ba, tpr, tnr = ba_curve(te, ho, grid, pi)
    if smooth > 1:                                   # lam muot de tranh bam dinh nhieu
        k = np.ones(smooth) / smooth
        ba = np.convolve(ba, k, mode="same")
        ba[:smooth] = ba[smooth]; ba[-smooth:] = ba[-smooth-1]
    i = int(np.argmax(ba))
    return grid[i], pi, float(ba[i]), float((te >= grid[i]).mean())
