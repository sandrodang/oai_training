#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
until grep -q "EXPLORE DONE" logs/explore.log; do sleep 15; done
run () { local n="$1"; shift
  local r=$(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --w_noise 0.30 --save_ckpt 0 "$@" 2>&1 | grep 'OOF over'); echo "$n | ${r:-CRASHED}"; }
echo "baseline: FGM eps=1.0 -> 0.7277"
run fgm15      --fgm 1.5
run fgm20      --fgm 2.0
run gceN_fgm   --fgm 1.0 --loss_n gce --gce_q 0.7
echo "=== EXPLORE2 DONE ==="
