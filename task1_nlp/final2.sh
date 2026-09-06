#!/bin/bash
cd /home/namdp36/oai/work
export CUDA_VISIBLE_DEVICES=7 HF_HOME=/home/namdp36/oai/work/hf TOKENIZERS_PARALLELISM=false
KIT=/home/namdp36/oai/ThiChinhThucData/NLP_Data/Participant_Kit
until grep -q "OOF over" logs/visobert_cons.log; do sleep 15; done
echo "=== cons 5-fold done ==="; grep -E "fold [0-9]\]|OOF over" logs/visobert_cons.log
echo "=== reference: visobert_fgm OOF = 0.7139 ==="
echo "=== ensembling fgm + cons ==="
./.venv/bin/python src/ensemble.py --npz oof/visobert_fgm_oof.npz oof/visobert_cons_oof.npz \
  --out oof/ens_final.npz 2>&1 | tail -12
echo "=== building v8 ==="
./.venv/bin/python src/make_sub.py --ver v8_fgm_cons --npz oof/ens_final.npz
./.venv/bin/python "$KIT/check_submission.py" --test "$KIT/data/public_test.zip" \
  --submission sub/v8_fgm_cons/FPTU_Promt_Engineer_task1_pub.zip
echo "=== FINAL2 DONE ==="
