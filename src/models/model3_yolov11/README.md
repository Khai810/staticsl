# YOLO11n ASL của Bảo

Nhánh RGB YOLO11n được đặt trong thư mục model 3 có sẵn của repo nhóm. Đây là một module độc lập với môi trường Python và cấu hình riêng. Các file của MLP, ST-Hybrid, notebook tổng hợp, dữ liệu và checkpoint chung giữ nguyên.

## Kết quả đã xác minh

Baseline `20261114T065750Z_baseline_d5cb4953` đã hoàn tất **50/50 epoch trên Tesla T4**, khoảng 1 giờ 59 phút. Dùng **9.042 train / 3.909 validation**, 29 đầu ra và 28 lớp có positive validation. Có học chuyển giao từ YOLO11n COCO; fine-tune toàn bộ tham số có thể học, seed 42, imgsz 320, batch 8.

| Chỉ số của checkpoint epoch 45 | Kết quả |
|---|---:|
| Accuracy mức ảnh tại confidence 0,25 | 98,26% |
| Macro F1 trên 28 lớp có support | 98,67% |
| Weighted F1 | 98,70% |
| mAP@0.5 | 96,30% |
| mAP@0.5:0.95 | 68,97% |

[Xem kết quả và biểu đồ](reports/baseline_20261114/RESULTS.md). Checkpoint Macro F1 ở **epoch 45**, khác native `best.pt` theo detection fitness ở **epoch 49**. Sweep chỉ validation chọn confidence **0,05**: Accuracy 98,72%, Macro F1 98,79%. Các kết quả này là phát triển trên validation, chưa có final test/người mới hoặc benchmark. `nothing` không có positive validation; box GT tự động là weak labels; metadata signer/session chưa đủ để kết luận signer-independent.

## Cấu trúc

```text
src/asl_yolo/train_release/     Train, validator, metrics, checkpoint journal và resume
src/asl_yolo/                   Các helper và module YOLO được bảo toàn từ source đã train
scripts/train_release.py        Điểm vào CLI
configs/                       Cấu hình auto V2 đã khóa
notebooks/03_*.ipynb            Colab mount Drive, khôi phục bundle và chạy từng giai đoạn
tests/test_training_release.py  13 kiểm thử phần mềm; không phải kết quả chất lượng model
reports/baseline_20261114/      Số liệu tổng hợp và biểu đồ thật, không có ảnh người tham gia
reports/asl_yolo11n_auto_v2_training_20261113/  Protocol và hướng dẫn chi tiết
ARTIFACTS.json                 Link Drive và SHA của dataset/source/model
PUBLISH_MANIFEST.json          Nguồn file, checksum và phạm vi thay đổi
```

## Chạy Colab

Mở [notebook đã chạy thành công](https://colab.research.google.com/drive/18S-EHozFcFV6deWvNv5Iy7SqjAdSD0kl), hoặc tải notebook 03 trong thư mục này lên Colab. Chọn GPU, mount Drive và cấp quyền cho notebook. Drive cần có thư mục `MyDrive/ASL_YOLO11n/exports/auto_v2_training_20261114` chứa bundle, source và ba phần dataset; [thư mục bundle](https://drive.google.com/drive/folders/1dJ9Zs-B-3CIRkxtrf0L8WjD6_pbg1ljZ) cần quyền truy cập của chủ sở hữu.

Notebook 03 khôi phục **source.zip đã khóa trên Drive** và dataset theo checksum rồi chạy ở `/content/asl_auto_v2_training`. Nó không tự lấy thay đổi mới từ Git. Toàn bộ 27 file source được khóa trong run gốc khớp byte với bản trong thư mục Git này. Nếu thay code cho thí nghiệm mới, cần đóng gói lại source và cập nhật bundle trước khi chạy. Không chạy lại 50 epoch chỉ để xem kết quả; xem reports hoặc run Drive đã hoàn tất.

## Chạy CLI từ đúng thư mục module

Từ root repo nhóm, chuyển vào thư mục này trước khi chạy lệnh:

```powershell
cd src/models/model3_yolov11
python scripts/train_release.py --help
python -m pytest -c pyproject.toml tests/test_training_release.py -q
```

Thư viện của nhánh được pin trong `requirements-training-colab.txt`. Colab đã có torch/torchvision CUDA; notebook giữ bộ CUDA này bằng constraints khi cài dependencies. Với máy riêng, tạo môi trường Python riêng và cài bộ PyTorch CUDA phù hợp trước khi cài requirements. Không cài chồng dependencies của nhánh vào môi trường TensorFlow chung.

Sau khi khôi phục đúng dataset và asset từ bundle vào root module:

```powershell
python scripts/train_release.py preflight --data-only
# Các lệnh dưới yêu cầu CUDA và tạo run mới, không tự chạy khi import module
python scripts/train_release.py train --mode smoke
python scripts/train_release.py resume-probe
python scripts/train_release.py train --mode baseline
```

Baseline chạy SGD, weight decay 0,0005, warmup 3 epoch và lịch learning rate tuyến tính; không L1 tường minh, early stopping tắt (`patience=0`). Checkpoint và predictions lưu mỗi epoch. Chọn model theo Macro F1 tại confidence 0,25, rồi Accuracy, rồi epoch sớm hơn. Giữ `allow_final_test=false`.

## Model và resume

[Model Macro F1 epoch 45](https://drive.google.com/file/d/1XMm9vNQjvxFftTAfh04U-vrO23hB8WPL/view) và [run đầy đủ](https://drive.google.com/drive/folders/1Ho7K2qwRJHOyIqUcVIzDgNUEr5CKGaKm) nằm trên Drive; SHA và mapping trong `ARTIFACTS.json` và `reports/baseline_20261114/`. Không thay checkpoint YOLO lịch sử trong `checkpoints/` chung bằng model mới một cách âm thầm.

`best_macro_f1.pt` dùng inference; `last_resume.pt` hoặc generation `resume.pt` mới giữ optimizer/scheduler/EMA/RNG để tiếp tục run **chưa hoàn tất**. Không dùng weight inference như checkpoint resume. Run 50 epoch này đã hoàn tất nên không resume nó để train thêm 100 epoch.

```powershell
# Chỉ áp dụng cho RUN chưa hoàn tất, đã xác minh process cũ dừng
python scripts/train_release.py --config RUN/config.json status --run RUN
python scripts/train_release.py --config RUN/config.json resume --run RUN --checkpoint RUN/checkpoints/last_resume.pt
python scripts/train_release.py --config RUN/config.json analyze --run RUN
python scripts/train_release.py --config RUN/config.json plots --run RUN
```

Inference cần class-aware NMS, IoU 0,7, `multi_label=True` và quy tắc box confidence cao nhất như validator. Không mặc định coi `YOLO.predict` với các tham số mặc định là cùng phép hậu xử lý. `inference_review.json` là khóa ngưỡng của lượt review validation; run Drive gốc không bị sửa.

[Hướng dẫn code và checkpoint chi tiết](reports/asl_yolo11n_auto_v2_training_20261113/CODE_GUIDE.md). Các tài liệu ngày 13/11 giữ lịch sử trước Colab; trạng thái hoàn tất hiện tại nằm ở phần kết quả ngày 14/11 trong thư mục này.
