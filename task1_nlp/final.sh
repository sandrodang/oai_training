#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
KIT=/home/namdp36/oai/ThiChinhThucData/NLP_Data/Participant_Kit
until grep -q "OOF over" logs/fgm_s7.log; do sleep 15; done
echo "=== seed-7 done ==="; grep -E "fold [0-9]\]|OOF over" logs/fgm_s7.log

echo "=== ensembling the two FGM seeds (weights fitted on OOF, honesty-guarded) ==="
./.venv/bin/python src/ensemble.py --npz oof/visobert_fgm_oof.npz oof/visobert_fgm_s7_oof.npz \
  --out oof/ens_fgm2.npz 2>&1 | tail -12

echo "=== building v8 ==="
./.venv/bin/python src/make_sub.py --ver v8_fgm_2seed --npz oof/ens_fgm2.npz
./.venv/bin/python "$KIT/check_submission.py" --test "$KIT/data/public_test.zip" \
  --submission sub/v8_fgm_2seed/FPTU_Promt_Engineer_task1_pub.zip
echo "=== FINAL DONE ==="
