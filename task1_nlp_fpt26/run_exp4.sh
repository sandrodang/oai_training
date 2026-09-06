#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
run () { local n="$1"; shift
  echo "$n | $(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --save_ckpt 0 "$@" 2>&1 | grep 'OOF over')"; }
# complete the FGM x w_noise grid (batch 2 changed both at once)
run fgm10_wn015   --fgm 1.0 --w_noise 0.15
run fgm15_wn030   --fgm 1.5 --w_noise 0.30
run fgm20_wn030   --fgm 2.0 --w_noise 0.30
# augmentation, always on top of the FGM winner
run aug03_fgm10   --fgm 1.0 --w_noise 0.30 --aug 0.3
run aug05_fgm10   --fgm 1.0 --w_noise 0.30 --aug 0.5
