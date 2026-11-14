# Hướng dẫn đọc và chạy code training auto V2

Đây là luồng mới, độc lập prototype 24 chữ. Release giữ nguyên; chỉ cấu hình training mới bật `allow_training=true`. Final test luôn tắt. Mở `IMPLEMENTATION_PLAN.md` và `SKELETON.md` trước; skeleton nguyên file còn ở `skeleton_snapshot/`.

## Thứ tự đọc

1. `configs/asl_yolo11n_auto_v2_training.json`: release/hash, các cổng, augmentation tham chiếu, optimizer/scheduler và confidence grid. Chỉ `runs_root` phụ thuộc runtime; batch chỉ giảm trong attempt mới khi CUDA OOM. Thay protocol/seed/imgsz/augmentation cần thiết kế thí nghiệm mới, không resume run cũ.
2. `src/asl_yolo/train_release/cli.py`: các lệnh và timeline. Không có lệnh final test, external hay webcam.
3. `config.py` → `dataset.py`: xác minh fingerprint/hash/checksum/mapping, subset đã chấp nhận, không còn pHash candidate cắt split, so nhóm hash/pixel/session đã biết với metadata master. Master chỉ được đọc metadata cho audit, không đọc ảnh holdout chạy model. Unknown signer/session giữ nguyên unknown.
4. `runner.py` → `engine.py`: new run, resume, OOM, adapter trainer/validator và trạng thái. Baseline cần gate smoke và resume GPU thật với cùng code/release.
5. `storage.py`: immutable generation/checksum/commit marker, history, lock, recovery. Chỉ epoch đã commit mới xuất hiện trong history.
6. `metrics.py`: toán mức ảnh/background, khác detection mAP. `reporting.py`: khóa model, val sweep, figures/gallery/report. `benchmark.py`: forward GPU tùy chọn sau khi có model.
7. `tests/test_training_release.py`: fixture nhân tạo chỉ dùng cho test phần mềm, không đi vào báo cáo kết quả model.

## Một batch đi qua đâu

Release chứa ảnh gốc giữ tỷ lệ và YOLO normalized xywh. Adapter runtime hardlink/copy byte-for-byte vào `.cache/train_release/<fingerprint>/...`; cache/label scan của Ultralytics không ghi vào release. Dataset train dùng toàn augmentation baseline đã khóa: RandomPerspective chỉ xoay ±12°/scale .07; translate, flip, mosaic, mixup, HSV, perspective, shear, bgr, cutmix/copy-paste đều tắt. `augmentations=[]` chặn Albumentations mặc định nếu runtime có cài package đó. Validation chỉ resize/letterbox/format, không augmentation.

Trainer tạo detector 29 đầu ra từ asset COCO YOLO11n. `pretrained_transfer.json` ghi tensor thực nạp; `layers.json` ghi từng parameter trainable/frozen sau setup. Freeze0 fine-tune các tham số học được, trừ DFL cố định của framework. Batch RGB → forward/loss box/cls/DFL → backward và kiểm tra gradient hữu hạn → SGD/EMA. Validator dùng EMA float32, NMS class-aware `multi_label=True`, IoU .7/max_det300/conf floor .001; detection metrics dùng toàn bộ boxes. Adapter lưu **top2 boxes sau NMS** mỗi ảnh: đủ tái tính highest-confidence classification ở mọi threshold và cờ nhiều hơn một box được chấp nhận, giảm dung lượng lưu mỗi epoch. Không cần inference val lần hai để chọn checkpoint.

Với positive GT, box cao nhất đạt threshold=.25 quyết định nhãn; nếu không có là NO_DETECTION, vẫn tính sai và giữ mẫu số. F1/recall nothing positive val support0=N/A, số FP riêng. Background label rỗng không biến thành class nothing; false positives trên background báo riêng và vẫn có trong detection evaluation.

## Mỗi epoch lưu gì

`framework/weights/{last.pt,best.pt,epoch*.pt}` giữ native fitness; final_eval có thể strip optimizer last/best. `checkpoints/epoch_NNNN_<generation>/resume.pt` giữ raw model float32, EMA float32, optimizer full dtype, scheduler, scaler, RNG Python/NumPy/torch/CUDA và loader generator. Metric macro F1 đo EMA float32 nên `best_macro_f1.pt` cũng giữ EMA float32; file này chỉ dùng inference. Chỉ `resume.pt` hoặc alias `last_resume.pt` dùng tiếp tục train.

Checkpoint/metrics/predictions được ghi tạm, fsync, đọc đối chiếu checksum, rồi commit marker `commits/epoch_NNNN.json` được công bố cuối. Alias và history là bản dẫn xuất; journal immutable giữ generation trước. Không hứa rename Drive atomic. Nếu ngắt giữa các bước, `status`/`recover` phát hiện file dở/corrupt và phục hồi prefix epoch hợp lệ, giữ lại dữ liệu hỏng trong recovery để kiểm tra. Định kỳ mặc định **mỗi epoch**, không tự xóa checkpoint cũ. Reserve ban đầu 12 GiB/run; `baseline_estimate.json` sau smoke tính lại từ kích thước checkpoint thật. Đây không phải quota Drive đảm bảo: cần xem dung lượng Drive còn thực tế.

Epoch time là train+val+native save, không gồm journal commit; session_seconds đo toàn phiên. Total history là tổng các epoch đã commit; epoch dang dở và chuẩn bị/resume có thể thêm wall time ngoài history. Không coi final_eval/on_fit_epoch_end cuối phiên là epoch mới. History CSV/JSON được dựng từ journal nên không append trùng sau resume. Log Ultralytics chính vẫn ở terminal và framework/results.csv; session log lưu các output Python, các metrics/checkpoint không lệ thuộc log terminal.

## Lệnh Windows từ project root

```powershell
# Audit thật trên CPU, không bỏ qua kiểm tra dữ liệu
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py preflight --data-only
# Full preflight: sẽ trả mã 2 khi dữ liệu đạt nhưng chưa có CUDA
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py preflight

# Các lệnh sau cần PyTorch CUDA và storage bền vững
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py train --mode smoke
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py resume-probe
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py train --mode baseline

# Dùng config của chính RUN, đặc biệt khi attempt OOM đã giảm batch
$run = 'runs/auto_v2/REPLACE_WITH_RUN_ID'
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" status --run $run
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" resume --run $run --checkpoint "$run/checkpoints/last_resume.pt"
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" analyze --run $run
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" plots --run $run
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" benchmark --run $run
```

Nếu alias ghi dở, chọn đúng generation trả bởi `status`, thay `--checkpoint` bằng đường dẫn đó. Nếu marker/checksum hỏng:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_release.py --config "$run/config.json" recover --run $run
# Dùng checkpoint generation được recover trả về để resume.
```

Chỉ khi đã tự xác minh process GPU host cũ thật sự dừng, mới được dùng `unlock --run $run --old-job-stopped`. Process local còn tồn tại bị chặn; host khác không tự suy là đã chết từ mất kết nối trình duyệt. Resume không đổi config/release/mapping/code/phiên bản; run hoàn tất bị chặn. Chỉ phục hồi đến epoch hoàn tất gần nhất, epoch dang dở chạy lại. Không hứa bit-for-bit vì loader iterator/augmentation/kernel/hardware có thể khác dù đã lưu RNG. Warm start/kéo dài50 epoch/đổi cấu hình là thí nghiệm mới, chưa được tự động hóa ở đây.

## Colab và đóng gói

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_training_notebook.py
.\.venv\Scripts\python.exe -X utf8 scripts/package_training_release.py
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_training_release.py -q --basetemp .cache/training_tests
```

Bundle mới dưới `exports/asl_yolo11n_auto_v2_training_20261113/<UTC>/`: source.zip, dataset.zip, bundle.json. Dữ liệu chỉ accepted train/val và metadata master/pHash cho audit; không ảnh test/Bao/pending. Không chứa credentials hoặc ASL checkpoint cũ. Gói được xác minh SHA từng entry và SHA archive. `verification/package.json` chỉ vị trí/kích thước/hash mới nhất; không ghi đè gói lịch sử.

Mở `notebooks/03_asl_auto_v2_training.ipynb` trên Colab GPU. Upload ba file bundle vào thư mục Drive bạn chọn và điền BUNDLE_DIR. Notebook kiểm tra thực sự có gói/đúng SHA, cache vào `/content/asl_auto_v2_training`, đặt runs_root về Drive đã mount. Cài `requirements-training-colab.txt` kèm constraints giữ torch/torchvision hiện có; không dùng lock CPU Windows. Pin trực tiếp dựa trên bộ local đã đọc; **GPU Colab stack chưa được xác minh cho đến smoke/probe thật**. Notebook gọi subprocess CLI, từng phần rõ ràng; không lệ thuộc biến từ notebook cũ.

## Model/threshold và artifact

`analyze` chỉ nhận baseline hoàn tất, khóa SHA `best_macro_f1.pt` trước sweep. Sweep chỉ val .05..95 bước .05; chọn macro F1 supported rồi Accuracy rồi threshold cao hơn. `inference.json` bất biến lưu mapping/SHA/fingerprint/rule/multi_label. Khi triển khai sau này phải tái dùng validator hoặc NMS `multi_label=True` này; YOLO.predict mặc định có thể chọn nhãn đơn cho từng anchor, không tự coi là cùng postprocess.

`metrics/validation_fixed.json` giữ .25, `validation_selected.json` giữ ngưỡng chọn; `validation_sweep.csv/json`, predictions CSV/JSON, per_class.csv, confusion CSV, detection curves JSON, history CSV/JSON là nguồn vẽ. `plots` tái tạo PNG300/SVG chỉ từ nguồn này. Gallery val có các nhóm đúng/sai/no detection/FP background/high confidence wrong, nhóm không quan sát được ghi số0. `REPORT.md`, captions và artifact_checksums.json truy về đúng model/release. Không có model thật thì các lệnh này dừng, không tạo placeholder metrics.

Benchmark riêng chỉ forward YOLO batch1 trên GPU và tensor đã nạp: 20 warmup +500 lượt, mean/p95 và throughput=1000/mean_ms. Tách khỏi preprocess/NMS/MediaPipe/camera/display; chưa có model thì chưa đo. Final test và so sánh ba mô hình phải là tác vụ riêng với ground truth/phạm vi/danh sách test chung đã khóa.
