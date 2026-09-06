#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
run () {  # name, extra args
  local name="$1"; shift
  local out=$(./.venv/bin/python src/train_mtl.py --tag "$name" --folds 0 --epochs 8 --maxlen 96 --save_ckpt 0 "$@" 2>&1 | grep "OOF over")
  echo "$name | $out"
}
run wn015           --w_noise 0.15
run wn005           --w_noise 0.05
run fgm10           --w_noise 0.30 --fgm 1.0
run fgm05_wn015     --w_noise 0.15 --fgm 0.5
