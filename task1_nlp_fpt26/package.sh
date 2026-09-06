#!/bin/bash
# Build the post-verification archive required by the organisers:
#   "toàn bộ mã nguồn (huấn luyện và suy luận) cùng checkpoint đã dùng
#    để sinh ra kết quả đã nộp"
set -euo pipefail
cd /home/namdp36/oai/work
OUT=FPTU_Promt_Engineer_task1_submission_package
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[1/6] compliance audit (must pass before packaging)"
./.venv/bin/python compliance_check.py > "$OUT/COMPLIANCE_REPORT.txt" 2>&1 || {
  echo "!! compliance check FAILED - refusing to package"; cat "$OUT/COMPLIANCE_REPORT.txt"; exit 1; }
tail -1 "$OUT/COMPLIANCE_REPORT.txt"

echo "[2/6] source code (training + inference)"
mkdir -p "$OUT/src"
cp src/*.py "$OUT/src/"
cp VERIFY.sh run_private.sh package.sh compliance_check.py verify_reproduction.py \
   requirements.lock.txt REPRODUCE.md "$OUT/"
rm -f "$OUT/src/dl_models.py"          # scratch helper, not part of the pipeline

echo "[3/6] checkpoints actually used for submitted results"
mkdir -p "$OUT/ckpt"
for t in "$@"; do
  [ -d "ckpt/$t" ] || { echo "   !! missing ckpt/$t"; exit 1; }
  cp -r "ckpt/$t" "$OUT/ckpt/"; echo "   + $t ($(ls ckpt/$t | wc -l) folds)"
done

echo "[4/6] artefacts: OOF probabilities, mined rules, fold groups, logs"
mkdir -p "$OUT/oof" "$OUT/rules" "$OUT/logs" "$OUT/sub"
cp oof/*.npz oof/groups.npy "$OUT/oof/" 2>/dev/null || true
cp rules/*.json "$OUT/rules/"
cp logs/*.log "$OUT/logs/" 2>/dev/null || true
cp -r sub/v* "$OUT/sub/"

echo "[5/6] manifest with checksums"
{
  echo "# Manifest - FPTU_Promt_Engineer - Task 1"
  echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## input data (organiser-supplied, unmodified)"
  md5sum data/*.csv
  echo
  echo "## package contents"
  find "$OUT" -type f -exec md5sum {} \; | sort -k2
} > "$OUT/MANIFEST.txt"
echo "   $(grep -c . "$OUT/MANIFEST.txt") lines"

echo "[6/6] archive"
tar czf "$OUT.tar.gz" "$OUT"
du -sh "$OUT" "$OUT.tar.gz"
echo "DONE -> $OUT.tar.gz"
