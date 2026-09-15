"""
src/preprocessing/prepare_hybrid_dataset.py
==============================================
File RIÊNG cho Model 2 (Hybrid Transformer-CNN). Vì bộ ASL Alphabet Dataset
có ~87.000 ảnh, ta KHÔNG pre-compute augmentation hay lưu ảnh đã resize ra
đĩa (sẽ tốn dung lượng và mất tác dụng ngẫu nhiên của augmentation qua từng
epoch). Thay vào đó, script này chỉ làm nhiệm vụ "preprocessing nhẹ":

    1. Quét toàn bộ ảnh trong data/raw/asl_alphabet/<label>/*.jpg
    2. Ghi ra 1 manifest CSV duy nhất: [file_path, label]
    3. Dùng src/utils/dataset_split.py (DÙNG CHUNG với Model 1 & 3) để chia
       train/val/test theo tỉ lệ 80/10/10 (đúng theo bài báo), stratify theo nhãn.
    4. Lưu 3 file CSV riêng: train.csv, val.csv, test.csv

Việc RESIZE + AUGMENT + TẠO ẢNH DUAL-PATH (primary/auxiliary) được thực hiện
ON-THE-FLY bởi HybridASLDataset (src/models/model2_hybrid_transformer_cnn/dataset.py)
ngay trong lúc huấn luyện — đây là cách làm chuẩn cho dataset ảnh lớn.

Cách chạy (CLI):
    python -m src.preprocessing.prepare_hybrid_dataset \
        --input_dir data/raw/asl_alphabet \
        --output_dir data/hybrid_manifest \
        --test_size 0.1 --val_size 0.1
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from configs.config import PreprocessConfig
from src.utils.dataset_split import split_dataset


def build_manifest(input_dir: str, class_labels: list) -> pd.DataFrame:
    """
    Quét thư mục ảnh gốc, trả về DataFrame [file_path, label].

    Tham số:
        input_dir (str): thư mục ảnh gốc, cấu trúc <input_dir>/<label>/<anh>.jpg
        class_labels (list): danh sách nhãn hợp lệ (dùng để bỏ qua thư mục lạ
            và đảm bảo thứ tự lớp nhất quán với model.py).

    Trả về:
        pd.DataFrame với 2 cột: "file_path", "label".
    """
    input_root = Path(input_dir)
    records = []

    for label in class_labels:
        label_dir = input_root / label
        if not label_dir.is_dir():
            print(f"[CẢNH BÁO] Không tìm thấy thư mục cho nhãn '{label}': {label_dir}")
            continue
        image_files = sorted(
            [f for f in label_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        for img_path in image_files:
            records.append({"file_path": str(img_path), "label": label})

    if not records:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong {input_root}. "
            f"Cấu trúc mong đợi: {input_root}/<label>/<anh>.jpg với label thuộc {class_labels}"
        )

    return pd.DataFrame(records)


def _parse_args() -> argparse.Namespace:
    cfg = PreprocessConfig()
    parser = argparse.ArgumentParser(
        description="Tạo manifest CSV + chia train/val/test cho ASL Alphabet Dataset (Model 2)."
    )
    parser.add_argument("--input_dir", type=str, default=cfg.raw_images_dir,
                         help="Thư mục ảnh gốc, tổ chức theo <input_dir>/<label>/<anh>.jpg")
    parser.add_argument("--output_dir", type=str, default=cfg.hybrid_manifest_dir,
                         help="Thư mục lưu 3 file manifest train.csv/val.csv/test.csv")
    parser.add_argument("--test_size", type=float, default=cfg.split.test_size,
                         help="Tỉ lệ tập test trên tổng dataset (mặc định 0.1 theo bài báo: 80/10/10).")
    parser.add_argument("--val_size", type=float, default=cfg.split.val_size,
                         help="Tỉ lệ tập validation trên tổng dataset (mặc định 0.1).")
    parser.add_argument("--random_state", type=int, default=cfg.split.random_state,
                         help="Seed để tái lập kết quả chia tập.")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = PreprocessConfig()
    cfg.split.test_size = args.test_size
    cfg.split.val_size = args.val_size
    cfg.split.random_state = args.random_state

    df = build_manifest(args.input_dir, cfg.hybrid.hybrid_class_labels)
    print(f"[prepare_hybrid_dataset] Tổng số ảnh tìm thấy: {len(df)}")
    print(f"[prepare_hybrid_dataset] Phân bố theo nhãn:\n{df['label'].value_counts()}")

    # split_dataset() dùng CHUNG với Model 1 & Model 3 (src/utils/dataset_split.py).
    # DataFrame này KHÔNG có cột group_column (session_id) -> tự động dùng
    # stratified split theo nhãn (đúng yêu cầu bài báo: "stratified sampling").
    train_df, val_df, test_df = split_dataset(df, cfg.split)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[prepare_hybrid_dataset] Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"[prepare_hybrid_dataset] Đã lưu: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
