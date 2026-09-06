#!/bin/bash
# Giai nen private_test.zip (ma hoa AES) — can mat khau BTC phat o gio thu 6.
# Dung:  ./unzip_private.sh 'MAT_KHAU'
set -e
cd /home/namdp36/oai/cv
[ -z "$1" ] && { echo "Thieu mat khau. Dung: ./unzip_private.sh 'MAT_KHAU'"; exit 1; }
./.venv/bin/python - "$1" <<'PY'
import pyzipper, sys, os, shutil
pw = sys.argv[1].encode()
src = "../ThiChinhThucData/CV_Data/private_test/private_test.zip"
tmp = "_pt_tmp"
shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
with pyzipper.AESZipFile(src) as z:
    z.setpassword(pw)
    z.extractall(tmp)
csvs = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f == "test.csv"]
assert csvs, "khong tim thay test.csv"
root = os.path.dirname(csvs[0])
shutil.rmtree("private_test", ignore_errors=True)
shutil.move(root, "private_test")
shutil.rmtree(tmp, ignore_errors=True)
import csv as C
rows = list(C.DictReader(open("private_test/test.csv")))
miss = [r for r in rows if not os.path.exists(os.path.join("private_test", r["relative_path"]))]
print(f"  {len(rows)} mau | {sum(1 for _,_,f in os.walk('private_test') for _ in f)} file | thieu {len(miss)} anh")
import collections
print("  theo category:", dict(sorted(collections.Counter(r['category'] for r in rows).items())))
assert not miss
PY
echo "  OK — san sang chay predict"
