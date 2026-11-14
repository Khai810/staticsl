# Trạng thái triển khai training YOLO11n auto V2

**Cập nhật 14/11/2026:** đã xác minh run Colab `20261114T065750Z_baseline_d5cb4953` hoàn tất **50/50 epoch trên Tesla T4**, khoảng 2 giờ; smoke 3 và resume probe 4 cũng đã hoàn tất. Checkpoint Macro F1 chọn epoch 45: Accuracy 98,26%, Macro F1 98,67% tại confidence 0,25 trên validation. Đã kiểm tra checkpoint và tái tính metrics từ predictions thật; xem [báo cáo review mới](../asl_yolo11n_baseline_review_20261114/REVIEW.md). Benchmark và final test chưa chạy.

**Phần bên dưới là snapshot triển khai ngày 13/11, trước khi chạy Colab.** Trạng thái hiện hành nằm trong `verification/FINAL_STATUS.json`; bản JSON trước Colab được lưu riêng.

Đã tạo skeleton thật và lưu snapshot, sau đó triển khai code, CLI, checkpoint journal/resume, metrics, figures/report, notebook Colab và script đóng gói cho release auto V2. **Chưa train thật: 0 epoch, chưa có model ASL mới hoặc checkpoint resume GPU đã được xác minh.** Môi trường local là torch2.5.1+cpu, CUDA=False; không chạy baseline CPU.

## Phương pháp và dữ liệu

Release dùng mapping29 đầu ra A..Z, del, nothing, space; mốc 9.042 train/3.909 val, fingerprint `fbb9d072079f9a08d42e953d5927b232f64497f0d4d3d3e8e9309e045772efdf`. Review_kind=automated, human_review_claimed=false; nhãn nguồn/box tự động là weak labels, chưa chuyên gia ASL xác nhận. Source split5/3/2 là đề xuất, accepted run subset khóa riêng. Metadata release không bị sửa để bật training; cấu hình training mới bật theo yêu cầu người dùng.

Preflight thật và kết quả kiểm thử sẽ được ghi trong `verification/` và bảng trạng thái cuối bên dưới. Không sử dụng metrics lịch sử hoặc YOLO metrics gán sẵn từ notebook nhóm trưởng. Không gọi kiểm tra phần mềm bằng fixture là kết quả training. Pretrained COCO ở assets/yolo11n.pt chỉ là đầu vào, không phải model ASL mới.

## Protocol trước kết quả

YOLO11n COCO, imgsz320/batch8/seed42/freeze0/AMPFalse, train augmentation xoay±12°/scale.07 và đầy đủ YAML khóa; val không augmentation. Cấu hình Codex chọn trước run: SGD lr0.01/momentum.937/weight_decay.0005, scheduler linear lrf.01/warmup3, patience0. Mọi cấu hình framework resolve được lưu. Fine-tune các tham số có thể học, trừ DFL cố định; không gọi đây là chỉ train đầu phân loại.

Metric chọn checkpoint: image-level `val_macro_f1_supported` tại conf.25/NMSIoU.7/class-aware multi_labelTrue, trên28 lớp positive validation; tie Accuracy rồi epoch sớm hơn. nothing positive val support0 → F1/recallN/A, kèm số FP. NO_DETECTION giữ trong cột prediction và mẫu số positive, không là lớp GT macro. Background no_hand báo rejection/FPR riêng và vẫn nằm trong detection evaluation. Native best.pt giữ fitness riêng, best_macro_f1.pt dùng chọn macro F1. Sweep .05..95 chỉ val sau khóa SHA checkpoint; kết quả có cả model lẫn threshold chọn trên val là phát triển, chưa độc lập.

Checkpoint raw model/EMA/optimizer/scheduler/scaler/RNG và metrics/predictions commit sau mỗi epoch. last_resume.pt/generation khác inference-only best_macro_f1.pt. Journal và checksum giúp phát hiện ghi dở; không hứa Drive rename atomic hoặc tái lập bit-for-bit. Bảo vệ tới epoch hoàn tất gần nhất; trước epoch đầu chưa có checkpoint cứu.

## Phân biệt nguồn yêu cầu

Nhận xét thực của cô được trích từ đoạn góp ý đỏ C00000 trong DOCX: leakage/người mới, so sánh cùng test/lớp hoặc tách kết quả, M/N, Coverage/Accuracy và giới hạn Wilson. 3 seed/ablation/McNemar trực tiếp nhắm MLP/ST-Hybrid, chỉ đề xuất tiếp cho YOLO. Nhóm trưởng tham khảo best checkpoint/validation/history/plots300dpi của TensorFlow ST-Hybrid; không chép landmark augmentation hoặc nhánh fallback model random. Bảng chi tiết ở IMPLEMENTATION_PLAN.md và bằng chứng nguồn ở verification/feedback_extracted.json, sources_before.json.

## Giới hạn và bước tiếp theo

Smoke GPU3, probe GPU4(dừng1/resume2..4), baseline50 và validation model report chỉ chạy khi có runtime CUDA/storage bền vững. Cần chạy smoke/probe thật với phiên bản Colab đã pin trước baseline; code review và unit tests chưa thay thế xác minh GPU. Không có ETA baseline đo được trước smoke, không có Accuracy/F1/mAP model mới để báo cáo. Benchmark20+500 chưa thực hiện. Không có runtime Colab được kết nối trong inventory browser của tác vụ; gói Drive mới chưa được giả định là tồn tại.

Chống trùng không chứng minh signer-independent khi session chưa biết; Wilson trên frame tương quan có giới hạn; J/Z chỉ tư thế tĩnh. Detection mAP sau này chỉ phản ánh weak localization labels. Final test group_test/external Bao vẫn khóa và không chạy. Chưa train hai mô hình khác, không xếp hạng ba mô hình. Ưu tiên kiểm chứng độc lập người mới/hai điều kiện nền ánh sáng trong tác vụ riêng sau khi có ground truth đầy đủ và checkpoint/threshold/mapping bất biến.

Lệnh và cell cụ thể để tiếp tục: CODE_GUIDE.md và notebooks/03_asl_auto_v2_training.ipynb. Model mới sau khi chạy thành công sẽ nằm `runs/<run_id>/checkpoints/best_macro_f1.pt`; checkpoint resume là `checkpoints/last_resume.pt` hoặc generation journal được status trả về. Hiện chưa có các model đó.

## Bảng trạng thái xác minh

| Giai đoạn | Trạng thái thực tế |
|---|---|
| Kế hoạch/skeleton/code/CLI/notebook | Đã triển khai; snapshot skeleton giữ riêng |
| Tests | **82 passed**, toàn bộ suite repository; gồm 13 test mới |
| Data preflight | Đạt: checksum/mapping/count/label/ảnh nền/leakage, asset load và loader batch thật |
| GPU/storage preflight | Local storage write/readback đạt, khoảng125 GiB trống; CUDA=False nên runtime_ready=False |
| Smoke GPU3 | Chưa chạy, 0 epoch; lệnh thử bị chặn đúng trước training |
| Resume GPU thật | Chưa chạy; unit test journal/optimizer fixture đạt, không thay bài thử GPU |
| Baseline50 | Chưa chạy, 0 epoch; không model ASL mới |
| Validation model/figures học/threshold/gallery | Chưa chạy; không Accuracy/F1/mAP model mới |
| Biểu đồ dữ liệu | Đã xuất PNG300/SVG + CSV/JSON từ release thật |
| Benchmark20+500 | Chưa chạy vì chưa model/GPU |
| Final test/external | Khóa, không chạy |

Kết quả dữ liệu thật: train8.870 positive +172 background =9.042; val3.842 positive +67 background =3.909. nothing22 positive train/0 positive val. Kiểm tra leakage đối chiếu89.659 dòng metadata master; accepted12.951 ảnh không xung đột theo hash/pixel/known-session và candidate pHash khóa. Có12.931 accepted rows chưa biết session, nên không kết luận signer-independent. Không sửa release, module cũ hoặc snapshot nhóm trưởng; kiểm tra hash trước/sau được lưu.

Biểu đồ phân bố trong verification/local_preflight/data_figures dùng ảnh thật của subset release, phân biệt background với positive nothing, đã được xem kiểm tra. Nguồn kiểm tra cuối: preflight.json, loader_augmentation_audit.json, all_tests_verified.log, gpu_gates.json, immutable_sources_verified.json, syntax_checks.json. Source hash và danh sách package thực tế ở source_hashes_final.json/environment_final.json. Bộ đóng gói local và checksum archive được mô tả trong verification/package.json; chưa upload hoặc giả định đã hiện diện trên Drive.

Preflight trả exit2 là **cổng CUDA chặn**, không phải dữ liệu sai. Lượt test ban đầu lỗi thư mục temp sandbox, đã chạy lại với basetemp trong workspace. Một test phát hiện tên file tạm quá dài trên Windows; đã rút ngắn temp/recovery filename và kiểm tra lại. Hai lượt preflight đang tạo adapter tuần tự được dừng để thay bằng xử lý file độc lập đa luồng; lần đầy đủ sau đã đạt data_ready=True. Không có checkpoint giả được tạo trong runs; các torch fixture nằm trong cache pytest và chỉ kiểm tra phần mềm.

Nguồn kỹ thuật đối chiếu với code local8.4.178: [Ultralytics Train](https://docs.ultralytics.com/modes/train/), [Callbacks](https://docs.ultralytics.com/usage/callbacks/), [Detection metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/). Vẫn dùng YOLO11n dù tài liệu online minh họa model đời khác.

## Bộ chạy đã đóng gói local

Đã xác minh source.zip 4.96 MiB và dataset.zip 238.72 MiB, checksum từng entry và archive đạt. Có12951 ảnh train/val, không ảnh holdout; source chỉ có pretrained COCO. Thư mục phụ lục: `exports/asl_yolo11n_auto_v2_training_20261113/20261113T140639Z`. Copy source.zip, dataset.zip và bundle.json lên BUNDLE_DIR bạn chọn trong notebook03 để tiếp tục GPU. Gói chưa được upload/kiểm chứng trên Colab; CUDA smoke/resume/baseline vẫn chưa chạy.
