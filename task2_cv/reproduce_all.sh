#!/bin/bash
# TAI TAO TOAN BO TU DAU: dung moi truong -> huan luyen 5 nhanh -> sinh du doan -> doi chieu.
# Chay:  ./reproduce_all.sh            (day du, ~35 phut tren 1 GPU)
#        ./reproduce_all.sh --verify   (bo qua huan luyen, chi doi chieu tu checkpoint co san)
set -e
cd "$(dirname "$0")"
P=./.venv/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HOME="$PWD/hf"
VERIFY_ONLY=""; [ "$1" = "--verify" ] && VERIFY_ONLY=1

echo "== 0. Kiem tra du lieu =="
[ -d dataset_train/train/category_01 ] || { echo "THIEU dataset_train/train/ (giai nen dataset_train.zip vao thu muc nay)"; exit 1; }
for c in 01 02 03 04 05 06; do printf "   category_%s: %s anh\n" $c $(ls dataset_train/train/category_$c | wc -l); done

if [ -z "$VERIFY_ONLY" ]; then
echo "== 1. Moi truong =="
[ -d .venv ] || uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "== 2. Huan luyen 5 nhanh (seed 1337, chi dung anh train-normal cua BTC) =="
$P src/run_patchcore.py --model wrn50      --long 512 --bs 8 --tag wrn50_L512_h180
$P src/run_patchcore.py --model dinov2b    --long 518 --bs 4 --tag dinov2b_L518_h180
$P src/run_patchcore.py --model dinov2l_ml --long 518 --bs 4 --tag dinov2l_ml_L518
$P src/run_patchcore.py --model dinov2g_ml --long 518 --bs 2 --tag dinov2g_ml_L518
$P src/dinomaly.py --tag dinomalyL --enc dinov2l_ml --layers 6,8,10,12 --iters 3000 --enc-bs 4

echo "== 3. Dong bang cau hinh (p = 0.42, chon tu feedback public LB, freeze truoc private) =="
$P src/freeze.py --rule p --p 0.42 --maha --out ckpt/FINAL/config.json --models \
  wrn50_L512_h180:wrn50:512,dinov2b_L518_h180:dinov2b:518,dinov2l_ml_L518:dinov2l_ml:518,dinov2g_ml_L518:dinov2g_ml:518,dinomalyL:dinomaly:518
CFG=ckpt/FINAL/config.json
else
CFG=${CFG:-ckpt/BEST_81.2/config.json}
fi

echo "== 4. Sinh du doan tren public =="
$P src/predict.py --config "$CFG" --split public --name REPRO_PUB
$P src/validate_sub.py sub/REPRO_PUB/task2_public_output.csv --split public

echo "== 5. Doi chieu voi file da nop =="
REF=${REF:-submissions/08_broad5/task2_public_output.csv}
[ -f "$REF" ] || REF=sub/08_broad5/task2_public_output.csv
if diff -q "$REF" sub/REPRO_PUB/task2_public_output.csv >/dev/null; then
  echo "   PASS — tai tao TRUNG KHOP 100% tung byte voi file da nop (public 81.2/100)"
else
  N=$($P -c "
import csv,sys
a={r['sample_id']:r['label'] for r in csv.DictReader(open('$REF'))}
b={r['sample_id']:r['label'] for r in csv.DictReader(open('sub/REPRO_PUB/task2_public_output.csv'))}
print(sum(a[k]!=b[k] for k in a))")
  echo "   LECH $N/480 nhan."
  echo "   Nguyen nhan co the: GPU khac (tich luy bfloat16 khac nhau) hoac sai batch size."
  echo "   Config da ghi san batch dung cho tung nhanh — xem checkpoints/config.json."
fi
