#!/bin/bash
# Dong goi bo HAU KIEM (khac voi file nop!).
#   File nop  = ZIP chi chua task2_private_output.csv  -> do run_private.sh sinh ra
#   Bo hau kiem = ma nguon + checkpoint + cach tai lap -> file nay sinh ra, GIU LAI de xuat trinh
# Dung:  ./package_audit.sh [ten_sub_private] [--with-weights]
set -e
cd /home/namdp36/oai/cv
NAME=${1:-PRIVATE_FINAL}
WITH_W=""; [ "$2" = "--with-weights" ] && WITH_W=1
OUT=audit_FPTU_Promt_Engineer_task2
rm -rf "$OUT" && mkdir -p "$OUT"

echo "[1/6] ma nguon — CHI giu file nam trong cay phu thuoc cua pipeline..."
mkdir -p "$OUT"/src
# huan luyen: run_patchcore.py (4 nhanh PatchCore) + dinomaly.py (nhanh tai tao)
# suy luan  : predict.py -> common, features, patchcore, dinomaly, make_sub, submit
# tien ich  : freeze.py (dong bang cau hinh), validate_sub.py (kiem dinh dang)
for f in common.py features.py patchcore.py dinomaly.py run_patchcore.py \
         predict.py submit.py make_sub.py freeze.py validate_sub.py; do
  cp "src/$f" "$OUT"/src/
done
cp run_private.sh reproduce_all.sh package_audit.sh requirements.txt "$OUT"/ 2>/dev/null || true
# kiem tra khong con import nao tro ra ngoai bo da cat
( cd "$OUT"/src && for f in *.py; do
    grep -hoE "^from [a-z_]+ import" "$f" | sed 's/from //;s/ import//' | while read m; do
      [ -f "$m.py" ] || echo "   CANH BAO: $f import $m nhung $m.py khong co trong bo"
    done; done )

echo "[2/6] checkpoint da sinh ra ket qua nop..."
# ten phai la ckpt/<tag>/ vi common.py tim checkpoint o do
mkdir -p "$OUT"/ckpt
cp -r ckpt/BEST_81.2/*/ "$OUT"/ckpt/ 2>/dev/null || true
mkdir -p "$OUT"/ckpt/FINAL && cp ckpt/BEST_81.2/config.json "$OUT"/ckpt/FINAL/config.json
cp ckpt/BEST_81.2/config.json ckpt/BEST_81.2/README.md "$OUT"/ 2>/dev/null || true

echo "[3/6] cac file da nop..."
mkdir -p "$OUT"/submissions
for d in "sub/$NAME" sub/08_broad5; do
  [ -d "$d" ] && cp -r "$d" "$OUT"/submissions/ || true
done

echo "[4/6] manifest moi truong + seed..."
{
  echo "# Moi truong tai lap"
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "python: $(./.venv/bin/python -V 2>&1)"
  ./.venv/bin/python - <<'PY'
import torch, timm, numpy, sklearn, PIL, platform
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.version.cuda}")
print(f"timm: {timm.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"pillow: {PIL.__version__}")
print(f"platform: {platform.platform()}")
print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
PY
  echo "seed: 1337  (random, numpy.random, torch, torch.cuda; cudnn.deterministic=True, benchmark=False)"
  echo ""
  echo "# Pretrained weights su dung (de bai muc 5 cho phep)"
  echo "timm/wide_resnet50_2.tv2_in1k"
  echo "timm/vit_base_patch14_reg4_dinov2.lvd142m"
  echo "timm/vit_large_patch14_reg4_dinov2.lvd142m"
  echo "timm/vit_giant_patch14_reg4_dinov2.lvd142m"
} > "$OUT"/ENVIRONMENT.txt
uv pip freeze --python .venv/bin/python > "$OUT"/requirements.txt 2>/dev/null || \
  ./.venv/bin/python -m pip freeze > "$OUT"/requirements.txt 2>/dev/null || \
  printf 'torch\ntimm\nscikit-learn\npandas\npillow\nnumpy\n' > "$OUT"/requirements.txt

echo "[5/6] checksum toan ven..."
( cd "$OUT" && find submissions ckpt \( -name '*.csv' -o -name '*.json' \) -type f | sort | xargs sha256sum > CHECKSUMS.txt 2>/dev/null || true )

echo "[6/6] huong dan tai lap..."
cat > "$OUT"/REPRODUCE.md <<'MD'
# Tai lap ket qua da nop — Task 2 (CV), team FPTU_Promt_Engineer

## Ket qua
- Public best: 81.2 / 100  (macro Balanced Accuracy = 0.812)
- Private: xem submissions/PRIVATE_FINAL/

## Mo hinh
Rank-ensemble 5 nhanh, moi nhanh kem 1 nhanh Mahalanobis, nguong quantile chung p = 0.42.

| Nhanh | Loai | Backbone (pretrained, de bai cho phep) | batch | Tham so HOC tu train-normal |
|---|---|---|---|---|
| wrn50_L512_h180   | PatchCore | wide_resnet50_2 ImageNet layer2+3 | 8 | memory bank 30k patch (k-center greedy coreset) |
| dinov2b_L518_h180 | PatchCore | DINOv2-B/14 reg4                  | 4 | memory bank 30k patch |
| dinov2l_ml_L518   | PatchCore | DINOv2-L/14 reg4, gop 3 tang      | 4 | memory bank 30k patch |
| dinov2g_ml_L518   | PatchCore | DINOv2-G/14 reg4, gop 3 tang      | 2 | memory bank 30k patch |
| dinomalyL         | Tai tao dac trung | DINOv2-L/14 dong bang     | 4 | decoder transformer 4 block (HUAN LUYEN, AdamW+OneCycle) |

Anh resize GIU TI LE (khong center-crop). Diem anh = trung binh 1% patch co khoang cach lon nhat.
Gop ensemble = trung binh THU HANG trong noi bo tung category, roi cat o quantile 1-p.

## Cach chay lai
    # dat dataset_train/ , public_test/ (va private_test/) cua BTC vao thu muc goc bo nay
    ./reproduce_all.sh              # dung lai checkpoint TU DAU roi doi chieu
    ./reproduce_all.sh --verify     # chi doi chieu tu checkpoint co san
    # -> phai trung khop 100% tung byte voi submissions/08_broad5/task2_public_output.csv

LUU Y batch size: tich luy bfloat16 phu thuoc batch. Chay sai batch se lech nho va
co the lat nhan cua mau nam sat nguong (da quan sat: 1/480). Config da ghi san batch dung.

## Tuan thu quy dinh
- Chi dung anh train-normal do BTC phat lam du lieu that. Khong dung anh / nhan / output
  tu bat ky bo du lieu ngoai nao, khong dung canonical anomaly images.
- Pretrained weights: de bai muc 5 cho phep ro rang.
- Moi nhan sinh TU DONG tu pipeline. Khong gan nhan thu cong cho bat ky sample nao.
- Seed co dinh 1337 cho moi thu vien ngau nhien; cudnn.deterministic = True.
- Nguong p = 0.42 chon tu feedback TONG HOP cua public leaderboard (muc 6 cho phep),
  duoc FREEZE truoc khi private mo va khong fit lai tren private.
- Khong sua thu cong du lieu dau vao hay tep ket qua.

## File
- src/            toan bo ma nguon (huan luyen + suy luan)
- ckpt/           memory bank + tham so Mahalanobis + trong so decoder cho ca 6 category
- submissions/    file CSV/ZIP da nop
- ENVIRONMENT.txt phien ban thu vien, GPU, seed, danh sach pretrained weights
- CHECKSUMS.txt   sha256 de doi chieu toan ven
MD

SZ=$(du -sh "$OUT" | cut -f1)
tar czf "$OUT".tar.gz "$OUT"
echo
echo "XONG. Bo hau kiem: $OUT/  ($SZ)  va  $OUT.tar.gz"
echo "  -> GIU LAI, KHONG nop len he thong. He thong chi nhan ZIP chua dung 1 file CSV."
