"""Kiem tra submission dung Submission Contract cua BTC truoc khi nop."""
import sys, csv, argparse, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import read_test_csv

REQUIRED = ["sample_id", "category", "label"]

def validate(csv_path, split):
    errs = []
    ref = read_test_csv(split)
    ref_cat = {sid: cat for sid, cat, _ in ref}

    with open(csv_path, newline="", encoding="utf-8") as f:
        raw = f.read()
    if raw.startswith("﻿"):
        errs.append("file co BOM (phai la UTF-8 khong BOM)")
    rdr = csv.reader(raw.splitlines())
    rows = list(rdr)
    if not rows:
        return ["file rong"]
    if rows[0] != REQUIRED:
        errs.append(f"header sai: {rows[0]} != {REQUIRED}")
    body = rows[1:]
    for i, r in enumerate(body, start=2):
        if len(r) != 3:
            errs.append(f"dong {i}: co {len(r)} cot, phai co dung 3"); break

    ids = [r[0] for r in body if len(r) == 3]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup:
        errs.append(f"{len(dup)} sample_id bi trung, vd {dup[:3]}")
    missing = set(ref_cat) - set(ids)
    extra = set(ids) - set(ref_cat)
    if missing: errs.append(f"thieu {len(missing)} sample_id, vd {sorted(missing)[:3]}")
    if extra:   errs.append(f"thua {len(extra)} sample_id, vd {sorted(extra)[:3]}")

    bad_cat = [r[0] for r in body if len(r) == 3 and r[0] in ref_cat and r[1] != ref_cat[r[0]]]
    if bad_cat: errs.append(f"{len(bad_cat)} dong category khong khop test.csv, vd {bad_cat[:3]}")
    bad_lab = [r[0] for r in body if len(r) == 3 and r[2] not in ("0", "1")]
    if bad_lab: errs.append(f"{len(bad_lab)} dong label khong phai 0/1, vd {bad_lab[:3]}")

    n1 = sum(1 for r in body if len(r) == 3 and r[2] == "1")
    per = collections.Counter(r[1] for r in body if len(r) == 3 and r[2] == "1")
    print(f"[validate] {csv_path}")
    print(f"  {len(body)} dong / {len(ref)} sample trong test.csv")
    print(f"  du doan anomaly: {n1} ({n1/max(1,len(body)):.1%})  theo cat: {dict(sorted(per.items()))}")
    return errs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--split", default="public", choices=["public", "private"])
    a = ap.parse_args()
    e = validate(a.csv, a.split)
    if e:
        print("\n  KHONG HOP LE:"); [print("   -", x) for x in e]; sys.exit(1)
    print("  => HOP LE, san sang nop.")
