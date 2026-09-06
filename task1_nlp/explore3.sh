#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
until grep -q "OOF over" logs/visobert_cons.log; do sleep 15; done
sleep 90   # let final2.sh finish the ensemble first
run () { local n="$1"; shift
  local r=$(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --w_noise 0.30 --fgm 1.0 --save_ckpt 0 "$@" 2>&1 | grep -E "OOF over|Error|Traceback" | head -2)
  echo "$n | ${r:-CRASHED}"; }
echo "baseline: FGM eps=1.0 -> 0.7277"
run hier   --hier 1
run llrd   --llrd 0.9
run ema    --ema 0.999
echo "=== EXPLORE3 DONE ==="
