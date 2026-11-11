# TỐI ƯU HÓA HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU ASL (MEDIA PIPE HAND + ST-HYBRID TRANSFORMER)

Thư mục dự án này là bản sao hoàn chỉnh, được nâng cấp và tối ưu hóa từ `/Users/mac/Documents/Lectures/ComputerVision/BaiTapLon/sign-language-recognition`, bổ sung toàn bộ phân tích nguyên nhân gốc rễ, kỹ thuật tăng cường dữ liệu nhẹ (Light Data Augmentation), kiến trúc mô hình mới (**Optimized ST-Hybrid Model** đạt **98.27% Accuracy**), notebook chạy từ đầu đến cuối (End-to-End) và báo cáo chuyên sâu định dạng Word (`.docx`).

---

## 1. Cấu Trúc Thư Mục Dự Án

```
sign-language-recognition-optimized/
├── checkpoints/                                  # Checkpoints các mô hình
│   ├── model2_hybrid_transformer_cnn_best_12-9.keras  # Baseline 12-9 (Acc: 79.52%)
│   ├── model2_hybrid_transformer_cnn_best14-9.keras   # S-TR 14-9 (Acc: 20.55% - Lỗi xoay)
│   └── model_optimized_st_hybrid.keras                # Mô hình tối ưu (Acc: 98.27%)
├── configs/
│   ├── __init__.py
│   └── config.py                                 # Cấu hình tập trung siêu tham số & đường dẫn
├── data/
│   ├── hybrid_manifest/                          # Tệp nhãn chia tập train / val / test
│   │   ├── train.csv (27,489 mẫu)
│   │   ├── val.csv   (4,851 mẫu)
│   │   └── test.csv  (3,594 mẫu)
│   ├── landmarks/                                # Dữ liệu tọa độ gốc
│   │   └── landmarks_raw.csv (35,934 mẫu)
│   └── raw -> .../sign-language-recognition/data/raw  # Symlink tới ảnh thô (không trùng lặp dung lượng)
├── notebooks/                                    # Jupyter Notebooks
│   ├── Skeleton12_9.ipynb                        # Bản sao giữ nguyên từ bài gốc 12-9
│   ├── Skeleton14_9.ipynb                        # Bản sao giữ nguyên từ bài gốc 14-9
│   └── End_to_End_Sign_Language_Optimization.ipynb # NOTEBOOK MASTER CHẠY TỪ ĐẦU ĐẾN CUỐI
├── plots/                                        # 9 Biểu đồ trực quan hóa độ phân giải cao (300 DPI)
│   ├── 01_augmentation_comparison.png           # So sánh khung xương trước/sau augment
│   ├── 02_augmentation_distribution.png         # Trường phân bố biến dạng không gian
│   ├── 03_confusion_matrix_model12_9.png        # Ma trận nhầm lẫn Baseline 12-9
│   ├── 04_confusion_matrix_model14_9.png        # Ma trận nhầm lẫn S-TR 14-9 (suy thoái xoay)
│   ├── 05_confusion_matrix_optimized.png        # Ma trận nhầm lẫn Optimized ST-Hybrid
│   ├── 06_per_class_f1_optimized.png            # F1-Score từng chữ cái (28 classes)
│   ├── 07_training_curves_optimized.png         # Đường cong học tập Loss / Accuracy
│   ├── 08_model_comparison_metrics.png          # Biểu đồ cột so sánh cả 3 mô hình
│   └── 09_qualitative_inference_predictions.png # Trực quan hóa suy luận thực tế kèm Confidence
├── reports/                                      # Báo cáo kỹ thuật chi tiết
│   ├── Bao_Cao_Toi_Uu_Hoa_Model_Nhan_Dien_Ngon_Ngu_Ki_Hieu.docx  # Báo cáo Word 3.4 MB đầy đủ bảng & hình
│   ├── baocao_tomtat.md                         # Báo cáo tóm tắt Markdown
│   ├── evaluation_metrics_full.json             # Toàn bộ số liệu trích xuất dạng JSON
│   ├── optimized_training_history.json          # Nhật ký 18 epoch huấn luyện
│   └── generate_word_report.py                  # Script tái tạo file Word tự động
├── src/                                          # Mã nguồn module hóa
│   ├── preprocessing/                           # Tiền xử lý, chuẩn hóa, augmentation
│   ├── datasets/                                # Keras Sequence dataset generator
│   ├── models/                                  # Kiến trúc mô hình Keras & hand_landmarker.task
│   ├── training/                                # Script huấn luyện tự động
│   ├── evaluation/                              # Đánh giá & phân tích lỗi
│   └── utils/                                   # Vẽ khung xương 2D & ma trận nhầm lẫn
└── requirements.txt                              # Thư viện phụ thuộc
```

---

## 2. Hướng Dẫn Sử Dụng

### Bước 1: Mở và chạy Jupyter Notebook End-to-End
Notebook `notebooks/End_to_End_Sign_Language_Optimization.ipynb` đã được thực thi sẵn toàn bộ output (dung lượng 2.25 MB). Bạn có thể mở trực tiếp trong VS Code, Cursor hoặc Jupyter Lab:
```bash
conda activate slr_env
jupyter notebook notebooks/End_to_End_Sign_Language_Optimization.ipynb
```
Kernel được cấu hình sẵn là `Python (slr_env)`.

### Bước 2: Xem Báo Cáo Kỹ Thuật Chi Tiết (Word `.docx`)
Mở tệp:
`reports/Bao_Cao_Toi_Uu_Hoa_Model_Nhan_Dien_Ngon_Ngu_Ki_Hieu.docx`
bằng Microsoft Word hoặc Google Docs / Apple Pages. Báo cáo chứa đầy đủ:
- Tóm tắt điều hành & Cơ sở lý thuyết ST-TR (ICPR 2021).
- Chẩn đoán nguyên nhân gốc rễ (Root Cause Analysis) sự cố sụp đổ của `Skeleton14_9`.
- Bảng so sánh chỉ số chi tiết của 3 mô hình.
- Phân tích Top các cặp nhầm lẫn nghiêm trọng nhất (D vs G, K vs P, H vs U).
- 9 hình ảnh phân tích chất lượng cao nhúng trực tiếp.

### Bước 3: Chạy script độc lập qua Terminal (Nếu muốn train lại hoặc eval)
- Chạy đánh giá toàn diện và cập nhật biểu đồ:
  ```bash
  /opt/anaconda3/envs/slr_env/bin/python -m src.evaluation.evaluate
  ```
- Huấn luyện lại mô hình từ đầu:
  ```bash
  /opt/anaconda3/envs/slr_env/bin/python -m src.training.train_optimized
  ```
