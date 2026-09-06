# Task 2 (CV) — Nhat ky thi dau | FPTU_Promt_Engineer

Metric: `Score = 100 x macro Balanced Accuracy` tren 6 category.
Bang xep hang hien `SCORE / MAX x 100%`; trang ket qua tung luot hien diem THO.

## Cac luot nop public

| # | Cau hinh | p | Diem | Ghi chu |
|---|---|---|---|---|
| 1 | toan nhan 0 | — | **50.0** | Moc neo: TPR=0, TNR=1 => BA=0.5 chinh xac. Xac nhan format + thang diem |
| 2 | wrn50-512 + dinov2b-518 + maha | 0.34 | 74.6 | sweep p |
| 3 | " | 0.42 | **77.9** | dinh sweep |
| 4 | " | 0.50 | 76.0 | sweep p |
| 5 | " | 0.58 | 74.2 | sweep p |
| 6 | wrn50-512 + dinov2b-**770** + maha | 0.42 | 77.5 | do phan giai cao hon -> KHONG giup |
| 7 | + chuan hoa mien nen JPEG | 0.42 | 77.1 | KHONG giup (mat chi tiet defect) |
| 8 | nguong rieng tung category, co 60% | — | 76.3 | KHONG giup (overfit) |
| 9 | nguong rieng tung category, co 100% | — | 76.5 | KHONG giup |
| 10 | wrn50-512 + **dinov2l** da tang + maha | 0.42 | 77.5 | model to hon -> KHONG giup |
| 11 | **ensemble 4 model da dang** | 0.42 | **80.4** | +2.5 — giam phuong sai, khong phai model tot hon |
| 12 | **ensemble 6 model** (+ViT-Giant, +dinov2b-ml) | 0.42 | *dang cho* | |

## Bai hoc chinh
1. **Moi cai tien don le deu that bai; gop rong moi thang.** Cac cau hinh don le nam gon
   trong 76.3-77.9, tuc trong nhieu thong ke (SE ~ 1.3 diem voi 480 anh). Chon cau hinh
   "tot nhat theo public" la chon theo nhieu. Ensemble 6 nhanh doc lap moi tao buoc nhay that.
2. **Khong co ham muc tieu offline dang tin.** `AUC_proxy` (test vs holdout-normal) du doan
   sai huong 4/4 lan. Synthetic anomaly tu tao thi bao hoa (AUROC 0.957 trong khi AUC that ~0.86)
   vi lo tao ra de hon defect that. => KHONG chay random/beam search: search tren ham muc tieu
   hong chi giup overfit nhanh hon.
3. **Nguong quantile tren test bat bien voi scale score, KHONG bat bien voi ti le anomaly.**
   `BA = (TPR+TNR)/2` khong chua prevalence, nen nguong toi uu trong score-space doc lap voi
   prevalence, nhung *quantile* tuong ung thi khong. (mo phong: `src/sim_threshold*.py`)
4. **Bo uoc luong BA khu chap** du doan dung duong cong p toan cuc (sai so 0.86 diem tren 3 diem
   chua thay) NHUNG sai khi ap cho tung category — sai so tung category triet tieu nhau khi lay
   trung binh, nen phep kiem dinh toan cuc khong chung minh duoc dieu gi cho tung category.

## Nhung gi da thu va loai bo
do phan giai cao (768/770/1022) | chuan hoa mien nen JPEG | nguong rieng tung category |
model lon hon (ViT-L, ViT-G don le) | quet topq/pooling | augment bank bang lat anh
