"""Chon model KHONG ton submission.

AUC_proxy = P(score_test > score_holdout_normal).
Test = tron (1-pi) normal + pi anomaly, holdout = normal thuan, nen:
    AUC_proxy = (1-pi)*0.5 + pi*AUC_true   =>  AUC_proxy = 0.5 + pi*(AUC_true - 0.5)
pi la hang so cua tap test => xep hang model theo AUC_proxy TUONG DUONG xep hang theo AUC_true.
Chi dung train-holdout + test score, khong dung nhan nao.
"""
import os, sys, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATS
from submit import get_scores

def auc(pos, neg):
    a = np.concatenate([pos, neg])
    r = np.empty(len(a)); r[np.argsort(a, kind="stable")] = np.arange(len(a))
    # xu ly tie bang rank trung binh
    order = np.argsort(a, kind="stable"); s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j+1] == s[i]: j += 1
        if j > i: r[order[i:j+1]] = (i + j) / 2
        i = j + 1
    return (r[:len(pos)].sum() - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg))

def evaluate(tags, split="public", maha=False):
    sc = get_scores(tags, split, maha)
    per = {c: auc(te, ho) for c, (_, te, ho) in sc.items()}
    return per, float(np.mean(list(per.values())))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="danh sach tag, phay ngan cach")
    ap.add_argument("--split", default="public")
    a = ap.parse_args()
    all_tags = a.tags.split(",")
    combos = []
    for r in range(1, len(all_tags) + 1):
        combos += [list(c) for c in itertools.combinations(all_tags, r)]
    rows = []
    for c in combos:
        for m in (False, True):
            per, mean = evaluate(c, a.split, m)
            rows.append(("+".join(c) + ("+maha" if m else ""), mean, per))
    rows.sort(key=lambda x: -x[1])
    print(f"{'cau hinh':44s} {'mean':>6s} " + " ".join(f"{c[-2:]:>6s}" for c in CATS))
    for n, mean, per in rows:
        print(f"{n:44s} {mean:6.4f} " + " ".join(f"{per[c]:6.4f}" for c in CATS))
    print("\n(AUC_proxy cao hon = tot hon. 0.5 = khong phan biet duoc.)")
