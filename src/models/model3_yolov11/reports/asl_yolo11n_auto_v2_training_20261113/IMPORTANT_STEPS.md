# Các bước quan trọng training auto V2

Các mốc dùng UTC; giờ Việt Nam UTC+7. Không train CPU, không final test/external, không dùng metrics lịch sử cho run mới.

## Đọc nguồn và lưu skeleton trước triển khai

Đã đọc README, kế hoạch02/11, release/policy/validation auto V2, DOCX feedback, các module cũ, code TensorFlow nhóm trưởng và notebook/Colab scripts. Bảng truy nguồn quyết định trong IMPLEMENTATION_PLAN.md. Feedback được tách theo paragraph/màu và văn bản; DOCX không có comments.xml. Source SHA trước thay đổi trong verification/sources_before.json. Skeleton thật10 file có NotImplementedError, chụp lại nguyên file trong skeleton_snapshot rồi mới triển khai.

## Quyết định triển khai

Adapter mới đọc release.json trực tiếp, không ép qua prototype review. SGD/linear/warmup là lựa chọn Codex công bố trước kết quả. Validator tái dùng NMS đầy đủ và lưu top2 đủ replay image metric; journal tách full resume raw model/EMA khỏi inference checkpoint. Native last/best có thể strip optimizer, custom checkpoint không qua final_eval. Tắt native tự giảm batch OOM trong cùng attempt, giữ thử lỗi rồi tạo run mới từ COCO.

## Các lỗi thực tế và sửa

- Pytest dùng temp mặc định trong sandbox không tạo được thư mục: giữ log new_tests.log; chuyển --basetemp vào .cache của workspace.
- Test recovery tìm thấy Windows MAX_PATH do temp filename quá dài: rút ngắn filename tạm/recovery, không đổi dữ liệu hay model. Log new_tests_retry.log còn lưu.
- Hai preflight đầu đang chuẩn bị adapter tuần tự được ngắt có chủ đích để tăng tốc hardlink/checksum bằng8 thread; không train hoặc sửa release. Lần đầy đủ được lưu preflight_final.log và local_preflight/preflight.json, data_ready=True/runtime_ready=False.


## 2026-11-13T13:59:51.941327+00:00 — blocked

Lệnh: `python scripts/train_release.py preflight --output reports/asl_yolo11n_auto_v2_training_20261113/verification/local_preflight`

```json
{
  "data_ready": true,
  "runtime_ready": false,
  "fingerprint": "fbb9d072079f9a08d42e953d5927b232f64497f0d4d3d3e8e9309e045772efdf",
  "counts": {
    "val": 3909,
    "train": 9042
  },
  "positive_support": {
    "train": {
      "I": 292,
      "J": 370,
      "C": 414,
      "V": 295,
      "L": 344,
      "G": 464,
      "A": 394,
      "H": 337,
      "N": 225,
      "O": 396,
      "D": 379,
      "B": 271,
      "E": 410,
      "R": 266,
      "X": 207,
      "K": 269,
      "M": 231,
      "T": 263,
      "space": 277,
      "S": 350,
      "Q": 302,
      "F": 462,
      "W": 289,
      "P": 294,
      "Y": 247,
      "Z": 276,
      "del": 250,
      "U": 274,
      "nothing": 22
    },
    "val": {
      "J": 167,
      "O": 184,
      "A": 160,
      "E": 170,
      "W": 127,
      "del": 119,
      "D": 169,
      "K": 93,
      "P": 125,
      "L": 131,
      "F": 204,
      "T": 117,
      "G": 237,
      "R": 112,
      "I": 136,
      "M": 89,
      "X": 89,
      "Q": 143,
      "Z": 118,
      "S": 139,
      "V": 124,
      "H": 169,
      "C": 185,
      "B": 110,
      "Y": 82,
      "U": 106,
      "space": 128,
      "N": 109
    }
  },
  "background_counts": {
    "train": 172,
    "val": 67
  },
  "blockers": [
    "PyTorch has no CUDA GPU; smoke/resume/baseline not executed"
  ],
  "artifact": "D:\\chipDocuments\\studyFolder\\collegeStudy\\year4\\semester1\\cv\\project\\ASL_YOLO11n_Codex_Starter\\reports\\asl_yolo11n_auto_v2_training_20261113\\verification\\local_preflight"
}
```

## 2026-11-13T14:04:27.603355+00:00 — failed

Lệnh: `python scripts/train_release.py train --mode smoke`

```json
{
  "error": "RuntimeError: CUDA required. No CPU training fallback; run preflight --data-only locally"
}
```

## 2026-11-13T14:04:31.070216+00:00 — failed

Lệnh: `python scripts/train_release.py resume-probe`

```json
{
  "error": "RuntimeError: Actual GPU resume probe not run: CUDA unavailable"
}
```

## 2026-11-13T14:05:52.273090+00:00 — Kiểm chứng cuối trước đóng gói

Toàn bộ82 tests đạt (13 mới), notebook9 code cell parse/schema hợp lệ/output rỗng; data preflight đọc đủ12951 ảnh/labels, không corrupt; train/val background172/67. CUDA=False, smoke và resume-probe bị chặn đúng, chưa epoch/checkpoint ASL thật. File bằng chứng và trạng thái xem verification/FINAL_STATUS.json.

## 2026-11-13T14:09:17.886874+00:00 — Đóng gói và xác minh hoàn tất

Lệnh: `python scripts/package_training_release.py`. Source 5,199,559 bytes, data 250,319,862 bytes; SHA toàn archive và từng entry đạt. Dữ liệu gồm12951 ảnh train/val và12951 labels, không ảnh holdout. Chỉ weight pretrained COCO trong source, không model ASL giả/cũ. Gói local: `exports/asl_yolo11n_auto_v2_training_20261113/20261113T140639Z`; metadata/checksum ở `bundle.json`, `verification/package.json`, `bundle_verification.json`. Chưa upload Drive.

## 2026-11-14T06:47:27.718517+00:00 — Upload dataset lên Drive

Đã chia ZIP gốc thành3 phần dưới100MiB, kiểm SHA ghép đúng ZIP gốc, upload đủ3 phần và đọc lại metadata xác nhận kích thước. Source/bundle/notebook cập nhật cùng link Colab; notebook tự ghép và xác minh SHA từng phần/toàn ZIP trước giải nén, dùng file .partial để phục hồi cache. Test chức năng ghép/giải nén fixture, schema và cú pháp9 code cells đạt. Không thay release/split, không train trong tác vụ upload. Chi tiết IDs/kích thước trong verification/drive_upload_20261114.json.


## 2026-11-14T09:23:09.927927+00:00 — Kiểm tra kết quả Colab hoàn tất

Run 20261114T065750Z_baseline_d5cb4953 thực sự hoàn tất 50/50 epoch trên Tesla T4, session 7177,18 giây. Smoke 3 và resume probe 4 hoàn tất riêng. Đã đối chiếu 50 history rows/commit markers, checksum model được chọn và last resume; load model và predict một ảnh validation thật thành công. Chọn checkpoint epoch 45 theo Macro F1 98,67% tại confidence 0,25, Accuracy 98,26%; native detection best thuộc epoch 49. Tái tính đủ 3909 validation predictions đã lưu, sweep grid khai báo trước chọn 0,05; không inference toàn tập lại, không final test và không sửa release hay run Drive. Báo cáo/biểu đồ/gallery mới tại ../asl_yolo11n_baseline_review_20261114/REVIEW.md.
