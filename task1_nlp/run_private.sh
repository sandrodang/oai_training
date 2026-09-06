#!/bin/bash
# ============================================================================
#  PRIVATE ROUND  --  Olympic AI 2026, Task 1, team FPTU_Promt_Engineer
#
#      ./run_private.sh <private_test.zip> [zip-password]
#
#  Inference only, from checkpoints trained during the public phase.
#  Env overrides:  TAGS="visobert_fgm visobert_cons"   W="0.455 0.545"
#                  SAFE=1  -> single best model only (fastest, most conservative)
#  Every parameter was learned from the organisers' training data.
#  `id` is used solely to join predictions back to rows.
# ============================================================================
set -euo pipefail
cd /home/namdp36/oai/work
ZIP="${1:?usage: ./run_private.sh <private_test.zip> [password]}"
PW="${2:-}"
KIT=/home/namdp36/oai/ThiChinhThucData/NLP_Data/Participant_Kit
PY=./.venv/bin/python
export CUDA_VISIBLE_DEVICES="${GPU:-7}" HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false

if [ "${SAFE:-0}" = "1" ]; then TAGS="visobert_fgm"; W="1.0"; BIAS="oof/bias_fgm.npz"
else TAGS="${TAGS:-visobert_fgm visobert_cons}"; W="${W:-0.455 0.545}"; BIAS="${BIAS:-oof/bias_ens.npz}"; fi

echo "[1/5] extract  (AES-capable; system unzip cannot read PK 5.1)"
if [ -n "$PW" ]; then $PY src/extract.py --zip "$ZIP" --password "$PW" --out data
else                 $PY src/extract.py --zip "$ZIP" --out data; fi
test -f data/private_test.csv || { echo "!! data/private_test.csv missing"; exit 1; }
$PY -c "import pandas as pd;d=pd.read_csv('data/private_test.csv');print(f'      rows={len(d)} cols={list(d.columns)} dup_id={d.id.duplicated().sum()}')"

echo "[2/5] inference  (models: $TAGS)"
NPZ=""
for t in $TAGS; do
  [ -d "ckpt/$t" ] || { echo "!! ckpt/$t missing"; exit 1; }
  $PY src/predict.py --tag "$t" --test private_test --out "oof/priv_$t.npz"
  NPZ="$NPZ oof/priv_$t.npz"
done

echo "[3/5] combine + build submission"
if [ -n "$BIAS" ]; then
  $PY src/make_sub.py --ver private --phase private --test private_test --npz $NPZ --w $W --bias_npz "$BIAS"
else
  $PY src/make_sub.py --ver private --phase private --test private_test --npz $NPZ --w $W
fi

echo "[4/5] validate with the organisers' checker"
$PY "$KIT/check_submission.py" --test data/private_test.csv \
    --submission sub/private/FPTU_Promt_Engineer_task1_pri.zip

echo "[5/5] READY"
ls -la sub/private/; unzip -l sub/private/FPTU_Promt_Engineer_task1_pri.zip
