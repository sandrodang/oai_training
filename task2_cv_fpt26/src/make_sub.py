"""Sinh file nop: CSV dung contract + ZIP dung quy uoc ten."""
import os, sys, csv, zipfile, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_test_csv, SUB_DIR

def write_sub(labels, split, name, team="FPTU_Promt_Engineer"):
    """labels: dict sample_id -> 0/1. Tra ve (csv_path, zip_path)."""
    rows = read_test_csv(split)
    out = os.path.join(SUB_DIR, name); os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"task2_{split}_output.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sample_id", "category", "label"])
        for sid, cat, _ in rows:
            w.writerow([sid, cat, int(labels[sid])])
    zip_path = os.path.join(out, f"{team}_task2_{'pub' if split=='public' else 'pri'}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=os.path.basename(csv_path))
    return csv_path, zip_path

def labels_from_scores(scores, taus):
    """scores: dict cat -> (ids, s). taus: dict cat -> tau."""
    out = {}
    for cat, (ids, s) in scores.items():
        for i, sid in enumerate(ids):
            out[sid] = int(s[i] >= taus[cat])
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public", choices=["public", "private"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--team", default="FPTU_Promt_Engineer")
    ap.add_argument("--all-zero", action="store_true",
                    help="Moc neo leaderboard: BA=0.5 moi category => SCORE=50 chinh xac")
    a = ap.parse_args()
    if not a.all_zero:
        sys.exit("chi ho tro --all-zero o day; cac sub khac sinh tu predict.py")
    labels = {sid: 0 for sid, _, _ in read_test_csv(a.split)}
    c, z = write_sub(labels, a.split, a.name, a.team)
    print("CSV:", c); print("ZIP:", z)
