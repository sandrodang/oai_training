#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
KIT=/home/namdp36/oai/ThiChinhThucData/NLP_Data/Participant_Kit

until grep -q "OOF over" logs/visobert_fgm.log; do sleep 15; done
echo "=== 5-FOLD DONE ==="
grep -E "fold [0-9]\]|OOF over" logs/visobert_fgm.log

echo "=== building v7 ==="
./.venv/bin/python src/tune_prior.py --npz oof/visobert_fgm_oof.npz --out oof/bias_fgm.npz 2>&1 | tail -3
./.venv/bin/python src/make_sub.py --ver v7_visobert_fgm --npz oof/visobert_fgm_oof.npz --bias_npz oof/bias_fgm.npz 2>&1 | head -1
./.venv/bin/python "$KIT/check_submission.py" --test "$KIT/data/public_test.zip" \
  --submission sub/v7_visobert_fgm/FPTU_Promt_Engineer_task1_pub.zip

echo "=== resuming exploration ==="
run () { local n="$1"; shift
  echo "$n | $(./.venv/bin/python src/train_mtl.py --tag "$n" --folds 0 --epochs 8 --maxlen 96 \
        --w_noise 0.30 --fgm 1.0 --save_ckpt 0 "$@" 2>&1 | grep 'OOF over')"; }
run gce_fgm    --loss gce --gce_q 0.7
run aug03_fgm  --aug 0.3
run cons_fgm   --cons 0.5
echo "=== ALL DONE ==="
