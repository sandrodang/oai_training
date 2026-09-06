#!/bin/bash
# =============================================================================
#  FPTU_Promt_Engineer — Task 1 (R-ViHSD)
#  HẬU KIỂM MỘT LỆNH DUY NHẤT
#
#      ./VERIFY.sh  [đường-dẫn-thư-mục-data]
#
#  Mặc định đọc ./data. Thư mục cần có training_set.csv, validation_set.csv,
#  public_test.csv (giải nén trực tiếp từ file .zip của BTC).
#
#  Script tự dựng môi trường, chạy audit tuân thủ, rồi sinh lại dự đoán từ
#  checkpoint và so từng dòng với file đã nộp. Thoát mã 0 nếu tất cả đạt.
# =============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DATA_SRC="${1:-./data}"
PY=./.venv/bin/python
FAIL=0
line() { printf '%s\n' "-------------------------------------------------------------------"; }
step() { line; echo ">>> $*"; line; }

step "0/4  Môi trường"
if [ ! -x "$PY" ]; then
  echo "    chưa có .venv, đang dựng..."
  if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.10 .venv >/dev/null
      uv pip install --python $PY torch --torch-backend=cu124 >/dev/null
      uv pip install --python $PY -r requirements.lock.txt >/dev/null
  else
      python3 -m venv .venv >/dev/null
      $PY -m pip install -q --upgrade pip
      $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
      $PY -m pip install -q -r requirements.lock.txt
  fi
fi
$PY -c "import torch,transformers;print(f'    python  {__import__(\"platform\").python_version()}');print(f'    torch   {torch.__version__}  cuda={torch.cuda.is_available()}');print(f'    tfmrs   {transformers.__version__}')" || { echo "!! môi trường lỗi"; exit 1; }

step "1/4  Dữ liệu đầu vào (phải khớp bản BTC cấp, chưa qua chỉnh sửa)"
mkdir -p data
for f in training_set.csv validation_set.csv public_test.csv; do
  [ -f "data/$f" ] || cp "$DATA_SRC/$f" "data/$f" 2>/dev/null
done
cat > /tmp/_md5.expected <<'MD5'
d020d0b781352080f789f5ff738db10d  data/training_set.csv
725e19db6fdd2069ee4407135858dbda  data/validation_set.csv
6e5c0dd7a82ce386a49c19043d9e4282  data/public_test.csv
MD5
if md5sum -c /tmp/_md5.expected 2>/dev/null; then echo "    checksum khớp"; else
  echo "    !! checksum KHÔNG khớp hoặc thiếu file — truyền đường dẫn data: ./VERIFY.sh /path/to/data"; FAIL=1; fi

step "2/4  Audit tuân thủ (id / seed / dữ liệu ngoài / mô hình học máy thật)"
$PY compliance_check.py || FAIL=1

step "3/4  Tái lập kết quả đã nộp từ checkpoint"
declare -a CASES=(
  "visobert_fgm|sub/v7_visobert_fgm/task1_public_output.csv|oof/bias_fgm.npz"
  "visobert_clean|sub/v6_visobert_clean/task1_public_output.csv|oof/bias_clean.npz"
)
[ -d ckpt/visobert_cons ] && CASES+=("visobert_cons|sub/v8_fgm_cons/task1_public_output.csv|")
for c in "${CASES[@]}"; do
  IFS='|' read -r tag sub bias <<< "$c"
  [ -d "ckpt/$tag" ] && [ -f "$sub" ] || { echo "    (bỏ qua $tag — không có trong gói)"; continue; }
  echo "    --- $tag  ->  $sub"
  if [ -n "$bias" ]; then $PY verify_reproduction.py --tag "$tag" --sub "$sub" --bias "$bias" || FAIL=1
  else                    $PY verify_reproduction.py --tag "$tag" --sub "$sub" || FAIL=1; fi
done

step "4/4  Kết luận"
if [ "$FAIL" -eq 0 ]; then
  echo "    TẤT CẢ ĐẠT — mã nguồn và checkpoint tái tạo đúng kết quả đã nộp."
  exit 0
else
  echo "    CÓ MỤC KHÔNG ĐẠT — xem chi tiết phía trên."
  exit 1
fi
