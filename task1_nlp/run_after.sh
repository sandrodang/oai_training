#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
# wait for the production 5-fold to finish
until grep -q "OOF over" logs/visobert_fgm.log; do sleep 20; done
echo "=== production done, resuming exploration (sequential, full speed) ==="
run () { local n="$1"; shift
  echo "$n | $(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --w_noise 0.30 --fgm 1.0 --save_ckpt 0 "$@" 2>&1 | grep 'OOF over')"; }
run gce_fgm    --loss gce --gce_q 0.7
run aug03_fgm  --aug 0.3
run cons_fgm   --cons 0.5
