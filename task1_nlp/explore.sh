#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
run () { local n="$1"; shift
  local r=$(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --w_noise 0.30 --fgm 1.0 --save_ckpt 0 "$@" 2>&1 | grep 'OOF over')
  echo "$n | ${r:-CRASHED}"; }
echo "baseline to beat: FGM alone = 0.7277"
run gce_fgm    --loss gce --gce_q 0.7
run aug03_fgm  --aug 0.3
run cons_fgm   --cons 0.5
echo "=== EXPLORE DONE ==="
