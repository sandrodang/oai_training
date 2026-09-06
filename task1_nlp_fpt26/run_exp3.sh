#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
run () { local n="$1"; shift
  echo "$n | $(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --save_ckpt 0 --w_noise 0.15 "$@" 2>&1 | grep 'OOF over')"; }
run base_wn015                                     # reference, w_noise=0.15
run gce07        --loss gce --gce_q 0.7
run gce04        --loss gce --gce_q 0.4
run ls01         --loss ls  --ls 0.1
run cons05       --cons 0.5
run cons10       --cons 1.0
