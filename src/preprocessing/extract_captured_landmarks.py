#!/usr/bin/env python3
"""
Trích xuất Landmark cho Dataset Tự Chụp (custom_captured)
Hỗ trợ cấu trúc thư mục đa thành viên:
  data/raw/custom_captured/
    ├── Khai_dataset/
    ├── Viet_dataset/
    ├── Bao_dataset/
    ├── Tien_dataset/
    └── Dat_dataset/
Với quy chuẩn tên ảnh: {Label}_{Member}_{Index:02d}.jpg (ví dụ: A_Khai_01.jpg, A_Bao_01.jpg)
Sử dụng HandLandmarkExtractor (MediaPipe Tasks HandLandmarker) kết hợp Adaptive Border Padding (không crop).
"""

import os
import re
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

# Tự động nạp thư mục gốc của project vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import MediaPipeConfig, ASL_CLASSES
from src.preprocessing.extract_landmarks import HandLandmarkExtractor

VALID_CLASS_SET = set(ASL_CLASSES)

def normalize_label(raw_label: str) -> str:
    low = raw_label.strip().lower()
    if low in ('del', 'delete'):
        return 'del'
    if low == 'space':
        return 'space'
    if low in ('nothing', 'no_hand', 'other_hand', 'nohand', 'otherhand'):
        return 'nothing'
    up = raw_label.strip().upper()
    if up.startswith('DEL'):
        return 'del'
    if up.startswith('SPACE'):
        return 'space'
    if up.startswith('NOHAND') or up.startswith('OTHERHAND') or up.startswith('NOTHING'):
        return 'nothing'
    if len(up) >= 1 and up[0] in VALID_CLASS_SET:
        return up[0]
    return raw_label

def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất landmarks cho ảnh tự chụp đa thành viên")
    parser.add_argument("--input_dir", "-i", default="data/raw/custom_captured",
                        help="Thư mục chứa các dataset tự chụp (*_dataset)")
    parser.add_argument("--output_csv", "-o", default="data/landmarks/landmarks_captured.csv",
                        help="File CSV lưu trữ landmark đã trích xuất")
    parser.add_argument("--pad", type=int, default=50,
                        help="Số pixel padding viền để mở rộng ngữ cảnh cổ tay cho MediaPipe")
    return parser.parse_args()

def extract_with_adaptive_padding(extractor: HandLandmarkExtractor, img_bgr: np.ndarray, pad: int = 50):
    """
    Trích xuất 21 điểm landmarks (63 số float) trên toàn ảnh (không crop).
    Nếu trực tiếp không bắt được, áp dụng kỹ thuật padding viền để mở rộng trường nhìn cổ tay.
    """
    # 1. Thử trực tiếp
    res = extractor.extract(img_bgr)
    if res.success:
        return res.landmarks_flat.tolist()

    h, w, _ = img_bgr.shape

    # 2. Thử padding viền đen
    padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    res = extractor.extract(padded)
    cur_pad = pad

    # 3. Thử padding viền replicate
    if not res.success:
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        res = extractor.extract(padded)

    # 4. Thử padding mở rộng 80px
    if not res.success:
        cur_pad = pad + 30
        padded = cv2.copyMakeBorder(img_bgr, cur_pad, cur_pad, cur_pad, cur_pad, cv2.BORDER_REPLICATE)
        res = extractor.extract(padded)

    if not res.success:
        return None

    # Ánh xạ toạ độ từ ảnh padded về ảnh gốc
    h_pad, w_pad, _ = padded.shape
    raw_pad = res.landmarks_raw  # (21, 3)
    coords = []
    for pt in raw_pad:
        x_px = pt[0] * w_pad - cur_pad
        y_px = pt[1] * h_pad - cur_pad
        z_px = pt[2] * w_pad
        coords.extend([float(x_px / w), float(y_px / h), float(z_px / w)])
    return coords

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def process_captured_dataset(input_dir: str, output_csv: str, pad: int = 50):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Thư mục dataset không tồn tại: {input_dir}")

    cfg = MediaPipeConfig(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        min_hand_presence_confidence=0.2
    )
    extractor = HandLandmarkExtractor(cfg)

    items_to_process = []
    valid_exts = ('.jpg', '.jpeg', '.png')
    detected_dataset_dirs = []

    for entry in sorted(os.listdir(input_dir)):
        full_entry = os.path.join(input_dir, entry)
        if os.path.isdir(full_entry) and entry.endswith("_dataset"):
            detected_dataset_dirs.append(entry)

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = sorted([d for d in dirs if not d.startswith('.')], key=natural_key)
        for fname in sorted(files, key=natural_key):
            if fname.lower().endswith(valid_exts) and not fname.startswith('.'):
                fpath = os.path.join(root, fname)
                name_no_ext = fname.rsplit('.', 1)[0]
                parts = name_no_ext.split('_')

                # Xác định label chuẩn
                if len(parts) >= 2 and parts[0] != "":
                    label = normalize_label(parts[0])
                else:
                    label = normalize_label(os.path.basename(root))

                # Xác định member từ tên file {Label}_{Member}_{Index} hoặc tên thư mục {Member}_dataset
                member = None
                if len(parts) >= 3:
                    member = parts[1]
                elif len(parts) == 2 and not parts[1].isdigit():
                    member = parts[1]
                else:
                    rel_parts = os.path.relpath(fpath, input_dir).split(os.sep)
                    for rp in rel_parts:
                        if rp.endswith("_dataset"):
                            member = rp.replace("_dataset", "")
                            break
                if not member:
                    member = "Khai"

                items_to_process.append((fpath, fname, label, member))

    if not items_to_process:
        print(f"⚠️ Thư mục '{input_dir}' chưa có ảnh (.jpg, .jpeg, .png).")
        if os.path.exists(output_csv):
            print(f"[*] Tìm thấy file CSV landmark sẵn có: {output_csv}. Giữ nguyên file này.")
            return pd.read_csv(output_csv)
        return pd.DataFrame()

    records = []
    missed = []
    member_stats = {}

    print(f"Bắt đầu trích xuất landmarks từ: {input_dir}")
    if detected_dataset_dirs:
        print(f"Các thư mục thành viên (*_dataset): {', '.join(detected_dataset_dirs)}")
    print(f"Tổng số ảnh phát hiện: {len(items_to_process):,}")

    for fpath, fname, label, member in items_to_process:
        img = cv2.imread(fpath)
        if img is None:
            missed.append((fname, "Không đọc được ảnh"))
            continue

        coords = extract_with_adaptive_padding(extractor, img, pad=pad)
        if coords is None:
            missed.append((fname, "Không phát hiện bàn tay"))
            continue

        member_stats[member] = member_stats.get(member, 0) + 1

        row = {f"r{i}": coords[i] for i in range(63)}
        row["label"] = label
        row["source_file"] = fname
        row["member"] = member
        records.append(row)

    extractor.close()

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    total_imgs = len(df) + len(missed)
    print("=" * 65)
    print("KẾT QUẢ TRÍCH XUẤT LANDMARK CAPTURED DATASET:")
    print(f"  - Tổng số ảnh xử lý:           {total_imgs:,}")
    print(f"  - Trích xuất thành công:       {len(df):,} ({len(df)/max(1, total_imgs)*100:.2f}%)")
    print(f"  - Phân bố theo thành viên:     {member_stats}")
    print(f"  - File CSV lưu tại:            {output_csv}")
    if missed:
        print(f"  - Ảnh không phát hiện tay ({len(missed)}): {[m[0] for m in missed[:5]]}...")
    else:
        print("  🎉 100% ẢNH ĐƯỢC TRÍCH XUẤT LANDMARKS THÀNH CÔNG!")
    print("=" * 65)
    return df

def main():
    args = parse_args()
    process_captured_dataset(args.input_dir, args.output_csv, pad=args.pad)

if __name__ == "__main__":
    main()
