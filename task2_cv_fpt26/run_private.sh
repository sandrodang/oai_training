#!/bin/bash
# Duong chay PRIVATE - mot lenh. Khong hoc gi, chi forward + nguong da freeze.
set -e
cd /home/namdp36/oai/cv
CFG=${1:-ckpt/BEST_81.2/config.json}
NAME=${2:-PRIVATE_FINAL}
ZIP=${3:-../ThiChinhThucData/CV_Data/private_test/private_test.zip}
export CUDA_VISIBLE_DEVICES=${GPU:-7} HF_HOME=/home/namdp36/oai/cv/hf

echo "[1/5] giai nen private_test..."
if [ ! -f private_test/test.csv ]; then
  rm -rf _pt_tmp && mkdir -p _pt_tmp && unzip -q -o "$ZIP" -d _pt_tmp
  # tu do layout: zip co the co hoac khong co thu muc bao ngoai
  SRC=$(dirname "$(find _pt_tmp -name test.csv | head -1)")
  [ -z "$SRC" ] && { echo "LOI: khong tim thay test.csv trong zip"; exit 1; }
  rm -rf private_test && mv "$SRC" private_test && rm -rf _pt_tmp
fi
echo "      test.csv: $(( $(wc -l < private_test/test.csv) - 1 )) mau"
echo "      anh tim thay: $(find private_test -name '*.jpg' | wc -l)"

echo "[2/5] kiem tra moi anh trong test.csv deu ton tai..."
./.venv/bin/python -c "
import csv,os,sys
miss=[r for r in csv.DictReader(open('private_test/test.csv'))
      if not os.path.exists(os.path.join('private_test', r['relative_path']))]
print(f'      thieu {len(miss)} anh'); sys.exit(1 if miss else 0)"

echo "[3/5] inference bang checkpoint da freeze ($CFG)..."
./.venv/bin/python src/predict.py --config "$CFG" --split private --name "$NAME"

echo "[4/5] kiem tra dinh dang..."
./.venv/bin/python src/validate_sub.py "sub/$NAME/task2_private_output.csv" --split private

echo "[5/5] XONG. Nop file nay:"
ls -la "sub/$NAME"/*.zip
