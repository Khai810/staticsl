"""
src/preprocessing/normalize_landmarks.py
===========================================
File DÙNG CHUNG cho cả 3 model (đúng tên & vai trò trong roadmap: "chuẩn hoá
theo cổ tay, scale bbox"). File này nhận ĐẦU VÀO là landmark THÔ đã được
extract_landmarks.py trích xuất, thực hiện:

    1. Chuẩn hoá toạ độ (dịch gốc theo cổ tay + chia scale) — dùng CHUNG
       cho Model 1 và Model 2.
    2. Với riêng Model 2: xử lý thêm phần ĐẶC THÙ CHUỖI THỜI GIAN — điền
       frame bị MediaPipe miss, cắt sliding window, pad/truncate về độ dài
       cố định (seq_len) — vì đây là bước "chuẩn hoá dữ liệu tuần tự" mô tả
       trong roadmap (Giai đoạn C, tuần 10: "Tiền xử lý: padding/truncate
       chuỗi... chuẩn hoá landmark như Model 1").

QUAN TRỌNG: Model 1 và Model 2 dùng chung hàm normalize_landmarks() bên
dưới để đảm bảo kết quả so sánh giữa 2 model là công bằng (apple-to-apple).

Cách chạy (CLI) — chuẩn hoá cho ẢNH TĨNH (Model 1):
    python -m src.preprocessing.normalize_landmarks \
        --mode images \
        --input_csv data/landmarks/landmarks_raw.csv \
        --output_csv data/landmarks/landmarks_normalized.csv

Cách chạy (CLI) — chuẩn hoá + ghép chuỗi cho VIDEO (Model 2):
    python -m src.preprocessing.normalize_landmarks \
        --mode videos \
        --input_dir data/landmarks/sequences_raw \
        --output_dir data/sequences \
        --seq_len 30 --stride 10 --use_sliding_window \
        --missing_frame_strategy interpolate
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.config import PreprocessConfig, NormalizeConfig, SequenceConfig
from src.preprocessing.extract_landmarks import FLAT_VECTOR_SIZE

# ============================================================================
# PHẦN 1 — CHUẨN HOÁ 1 LANDMARK (dùng chung cho Model 1 & Model 2)
# ============================================================================

def normalize_landmarks(landmarks_raw: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    """
    Chuẩn hoá 1 bộ landmark (21, 3) thành vector đã dịch gốc toạ độ + scale.

    Tham số:
        landmarks_raw (np.ndarray): shape (21, 3), toạ độ (x, y, z) thô lấy
                                     trực tiếp từ extract_landmarks.py.
        cfg (NormalizeConfig): cấu hình chuẩn hoá, gồm:
            - origin_index (int): landmark làm gốc toạ độ (mặc định 0 = cổ tay).
            - reference_pair (Tuple[int,int]): cặp landmark tính khoảng cách
              tham chiếu để chia scale (mặc định (0, 9)).
            - normalize_z (bool): có chuẩn hoá trục z hay không.
            - round_decimals (int): số chữ số thập phân làm tròn kết quả.

    Trả về:
        np.ndarray shape (63,) — vector đã chuẩn hoá.

    Công thức:
        1. Dịch gốc: p_i' = p_i - p_origin        (bất biến vị trí)
        2. Tính scale = khoảng_cách(p_a', p_b')    (a, b lấy từ reference_pair)
        3. Chia scale: p_i'' = p_i' / scale        (bất biến khoảng cách camera)
    """
    assert landmarks_raw.shape == (21, 3), (
        f"landmarks_raw phải có shape (21, 3), nhận được {landmarks_raw.shape}"
    )

    coords = landmarks_raw.copy()

    # Bước 1: dịch gốc toạ độ về landmark origin_index (mặc định: cổ tay).
    origin = coords[cfg.origin_index].copy()
    shifted = coords - origin

    # Bước 2 + 3: chia theo khoảng cách tham chiếu để bất biến scale.
    a_idx, b_idx = cfg.reference_pair
    scale = np.linalg.norm(shifted[a_idx] - shifted[b_idx])
    scale = scale if scale > 1e-6 else 1e-6  # tránh chia cho 0

    normalized = shifted / scale
    if not cfg.normalize_z:
        normalized[:, 2] = shifted[:, 2]  # z giữ nguyên độ lệch, không chia scale

    normalized = np.round(normalized, cfg.round_decimals)
    return normalized.flatten()  # shape (63,)


def normalize_flat_vector(flat_63: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    """Tiện ích: chuẩn hoá trực tiếp từ vector phẳng (63,) thay vì (21,3)."""
    return normalize_landmarks(flat_63.reshape(21, 3), cfg)


def normalize_batch(landmarks_batch: np.ndarray, cfg: NormalizeConfig) -> np.ndarray:
    """
    Chuẩn hoá HÀNG LOẠT — dùng khi xử lý toàn bộ 1 tập ảnh (Model 1) cùng lúc.

    Tham số:
        landmarks_batch (np.ndarray): shape (N, 21, 3).
        cfg (NormalizeConfig): giống hàm normalize_landmarks ở trên.

    Trả về:
        np.ndarray shape (N, 63).
    """
    return np.stack([normalize_landmarks(lm, cfg) for lm in landmarks_batch])


# ============================================================================
# PHẦN 2 — XỬ LÝ CHUỖI THỜI GIAN (riêng cho Model 2)
# ============================================================================

def fill_missing_frames(
    frames: List[Optional[np.ndarray]], cfg: SequenceConfig
) -> Optional[List[np.ndarray]]:
    """
    Xử lý các frame bị miss (giá trị None do MediaPipe không detect được tay)
    trong 1 chuỗi frame ĐÃ CHUẨN HOÁ.

    Tham số:
        frames (List[Optional[np.ndarray]]): danh sách vector (63,) đã chuẩn
            hoá theo thứ tự thời gian; phần tử = None nếu frame đó bị miss.
        cfg (SequenceConfig): cấu hình, các trường liên quan:
            - missing_frame_strategy (str): "zero" | "interpolate" | "drop_sample"
            - max_missing_ratio (float): tỉ lệ miss tối đa cho phép.

    Trả về:
        List[np.ndarray] đã điền đầy đủ (không còn None), hoặc None nếu mẫu
        này bị LOẠI BỎ (vượt quá max_missing_ratio, hoặc
        missing_frame_strategy="drop_sample" mà có ít nhất 1 frame miss).
    """
    n_missing = sum(1 for f in frames if f is None)
    missing_ratio = n_missing / len(frames) if frames else 1.0

    if cfg.missing_frame_strategy == "drop_sample" and n_missing > 0:
        return None
    if missing_ratio > cfg.max_missing_ratio:
        return None
    if n_missing == 0:
        return frames  # type: ignore

    if cfg.missing_frame_strategy == "zero":
        return [f if f is not None else np.zeros(FLAT_VECTOR_SIZE, dtype=np.float32) for f in frames]

    if cfg.missing_frame_strategy == "interpolate":
        return _linear_interpolate(frames)

    raise ValueError(f"missing_frame_strategy không hợp lệ: {cfg.missing_frame_strategy}")


def _linear_interpolate(frames: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    """
    Nội suy tuyến tính các frame None:
        - Đoạn None nằm GIỮA 2 frame hợp lệ -> nội suy tuyến tính giữa 2 frame đó.
        - Đoạn None nằm ở ĐẦU chuỗi (chưa có frame hợp lệ nào trước đó) ->
          gán bằng frame hợp lệ ĐẦU TIÊN tìm thấy (forward-fill).
        - Đoạn None nằm ở CUỐI chuỗi (không còn frame hợp lệ nào sau đó) ->
          gán bằng frame hợp lệ CUỐI CÙNG tìm thấy (backward-fill).
    """
    n = len(frames)
    result: List[Optional[np.ndarray]] = list(frames)

    valid_indices = [i for i, f in enumerate(result) if f is not None]
    if not valid_indices:
        return [np.zeros(FLAT_VECTOR_SIZE, dtype=np.float32) for _ in range(n)]

    first_valid_idx, last_valid_idx = valid_indices[0], valid_indices[-1]

    for i in range(first_valid_idx):
        result[i] = result[first_valid_idx]
    for i in range(last_valid_idx + 1, n):
        result[i] = result[last_valid_idx]

    i = first_valid_idx
    while i < last_valid_idx:
        if result[i + 1] is None:
            j = i + 1
            while result[j] is None:
                j += 1
            left, right = result[i], result[j]
            gap = j - i
            for k in range(i + 1, j):
                t = (k - i) / gap
                result[k] = (1 - t) * left + t * right
            i = j
        else:
            i += 1

    return result  # type: ignore


def pad_or_truncate(sequence: np.ndarray, cfg: SequenceConfig) -> np.ndarray:
    """
    Đưa 1 chuỗi về ĐỘ DÀI CỐ ĐỊNH cfg.seq_len.

    Tham số:
        sequence (np.ndarray): shape (T, 63), T là độ dài chuỗi gốc (bất kỳ).
        cfg (SequenceConfig): các trường liên quan:
            - seq_len (int): độ dài cố định mong muốn.
            - padding (str): "pre" | "post".
            - truncating (str): "pre" | "post".

    Trả về:
        np.ndarray shape (seq_len, 63).
    """
    T, D = sequence.shape
    if T == cfg.seq_len:
        return sequence

    if T > cfg.seq_len:
        if cfg.truncating == "pre":
            return sequence[T - cfg.seq_len:]
        elif cfg.truncating == "post":
            return sequence[: cfg.seq_len]
        raise ValueError(f"truncating không hợp lệ: {cfg.truncating}")

    pad_len = cfg.seq_len - T
    pad_block = np.zeros((pad_len, D), dtype=sequence.dtype)
    if cfg.padding == "pre":
        return np.concatenate([pad_block, sequence], axis=0)
    elif cfg.padding == "post":
        return np.concatenate([sequence, pad_block], axis=0)
    raise ValueError(f"padding không hợp lệ: {cfg.padding}")


def sliding_window_split(sequence: np.ndarray, cfg: SequenceConfig) -> List[np.ndarray]:
    """
    Cắt 1 video DÀI thành NHIỀU mẫu chuỗi ngắn bằng sliding window.

    Tham số:
        sequence (np.ndarray): shape (T, 63), T >= cfg.seq_len.
        cfg (SequenceConfig):
            - seq_len (int): độ dài mỗi cửa sổ.
            - stride (int): bước nhảy giữa 2 cửa sổ liên tiếp.

    Trả về:
        List[np.ndarray], mỗi phần tử shape (seq_len, 63).
    """
    T = sequence.shape[0]
    if T < cfg.seq_len:
        return [sequence]

    windows = []
    start = 0
    while start + cfg.seq_len <= T:
        windows.append(sequence[start: start + cfg.seq_len])
        start += cfg.stride
    return windows


def build_sequence_samples(
    frames: List[Optional[np.ndarray]], cfg: SequenceConfig
) -> List[np.ndarray]:
    """
    HÀM TỔNG HỢP cho Model 2 — pipeline đầy đủ cho 1 video/phiên quay:
        1. Điền frame miss (fill_missing_frames)
        2. Cắt sliding window (nếu cfg.use_sliding_window=True)
        3. Pad/truncate từng cửa sổ về đúng seq_len (pad_or_truncate)

    Tham số:
        frames: danh sách vector (63,) ĐÃ CHUẨN HOÁ hoặc None theo thứ tự
                thời gian.
        cfg: SequenceConfig.

    Trả về:
        List[np.ndarray], mỗi phần tử shape (cfg.seq_len, 63) — chính là các
        mẫu (X) sẽ lưu thành .npy để huấn luyện Model 2. Trả về list RỖNG
        nếu toàn bộ video bị loại bỏ do quá nhiều frame miss.
    """
    filled = fill_missing_frames(frames, cfg)
    if filled is None:
        return []

    sequence = np.stack(filled)  # shape (T, 63)

    if cfg.use_sliding_window and sequence.shape[0] > cfg.seq_len:
        windows = sliding_window_split(sequence, cfg)
    else:
        windows = [sequence]

    return [pad_or_truncate(w, cfg) for w in windows]


# ============================================================================
# PHẦN 3 — CLI: áp dụng chuẩn hoá lên toàn bộ dataset đã trích xuất
# ============================================================================

def normalize_image_dataset(input_csv: str, output_csv: str, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Đọc CSV landmark THÔ (do extract_landmarks.py --mode images tạo ra),
    áp dụng normalize_landmarks() cho từng dòng, lưu ra CSV đã chuẩn hoá
    — đây chính là dataset cuối cùng để huấn luyện Model 1 (MLP).

    Tham số:
        input_csv (str): file CSV có cột "r0".."r62", "label", "source_file".
        output_csv (str): file CSV đầu ra, cột "f0".."f62", "label", "source_file".
        cfg (PreprocessConfig): dùng cfg.normalize.
    """
    df_raw = pd.read_csv(input_csv)
    raw_cols = [c for c in df_raw.columns if c.startswith("r")]
    assert len(raw_cols) == FLAT_VECTOR_SIZE, (
        f"Kỳ vọng {FLAT_VECTOR_SIZE} cột landmark thô, tìm thấy {len(raw_cols)}."
    )

    normalized_rows = []
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="[normalize_landmarks] images"):
        flat = row[raw_cols].to_numpy(dtype=np.float32)
        norm_vec = normalize_flat_vector(flat, cfg.normalize)
        new_row = {f"f{i}": v for i, v in enumerate(norm_vec)}
        new_row["label"] = row["label"]
        new_row["source_file"] = row["source_file"]
        normalized_rows.append(new_row)

    df_norm = pd.DataFrame(normalized_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_norm.to_csv(output_csv, index=False)
    print(f"[normalize_landmarks] Đã lưu dataset chuẩn hoá (Model 1): {output_csv} — shape {df_norm.shape}")
    return df_norm


def normalize_video_dataset(
    input_dir: str, output_dir: str, labels_csv: str, cfg: PreprocessConfig
) -> pd.DataFrame:
    """
    Đọc toàn bộ file .npy landmark THÔ (do extract_landmarks.py --mode videos
    tạo ra, tên file "<label>__<session_id>.npy", có thể chứa NaN ở frame
    miss), thực hiện:
        1. Chuẩn hoá từng frame hợp lệ bằng normalize_landmarks().
        2. Gọi build_sequence_samples() để điền frame miss + cắt sliding
           window + pad/truncate về seq_len cố định.
        3. Lưu mỗi mẫu chuỗi thành 1 file .npy riêng + ghi CSV ánh xạ nhãn.

    Đây chính là dataset cuối cùng để huấn luyện Model 2 (1D-CNN + BiLSTM).

    Tham số:
        input_dir (str): thư mục chứa .npy thô từ extract_landmarks.py.
        output_dir (str): thư mục lưu .npy đã xử lý xong (sẵn sàng train).
        labels_csv (str): đường dẫn CSV lưu [file_name, label, session_id].
        cfg (PreprocessConfig): dùng cfg.normalize và cfg.sequence.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_root = Path(input_dir)
    raw_files = sorted(input_root.glob("*.npy"))
    if not raw_files:
        raise FileNotFoundError(f"Không tìm thấy file .npy nào trong {input_root}.")

    records = []
    sample_counter = 0

    for raw_path in tqdm(raw_files, desc="[normalize_landmarks] videos"):
        # Tên file dạng "<label>__<session_id>.npy" do extract_landmarks.py đặt.
        label, session_id = raw_path.stem.split("__", maxsplit=1)
        raw_sequence = np.load(raw_path)  # shape (T, 63), có thể chứa NaN

        frames_normalized: List[Optional[np.ndarray]] = []
        for flat_frame in raw_sequence:
            if np.isnan(flat_frame).any():
                frames_normalized.append(None)  # frame miss -> giữ None, xử lý ở build_sequence_samples
            else:
                frames_normalized.append(normalize_flat_vector(flat_frame, cfg.normalize))

        samples = build_sequence_samples(frames_normalized, cfg.sequence)
        if len(samples) == 0:
            print(f"[CẢNH BÁO] Bỏ qua {raw_path.name} — quá nhiều frame miss.")
            continue

        for sample in samples:
            file_name = f"{label}_{session_id}_{sample_counter:05d}.npy"
            np.save(os.path.join(output_dir, file_name), sample.astype(np.float32))
            records.append({"file_name": file_name, "label": label, "session_id": session_id})
            sample_counter += 1

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
    df.to_csv(labels_csv, index=False)

    print(f"\n[normalize_landmarks] Tổng số mẫu chuỗi tạo ra: {len(df)}")
    if len(df) > 0:
        print(f"[normalize_landmarks] Phân bố theo nhãn:\n{df['label'].value_counts()}")
    print(f"[normalize_landmarks] Đã lưu dataset (Model 2) vào: {output_dir}")
    print(f"[normalize_landmarks] Đã lưu file nhãn tại: {labels_csv}")
    return df


def _parse_args() -> argparse.Namespace:
    cfg = PreprocessConfig()
    parser = argparse.ArgumentParser(
        description="Chuẩn hoá landmark (+ ghép chuỗi cho Model 2) từ dữ liệu do extract_landmarks.py tạo ra."
    )
    parser.add_argument("--mode", type=str, choices=["images", "videos"], required=True,
                         help="'images' cho Model 1 (ảnh tĩnh); 'videos' cho Model 2 (chuỗi).")

    # --- images ---
    parser.add_argument("--input_csv", type=str, default=cfg.landmarks_csv_path,
                         help="[mode images] CSV landmark thô đầu vào (từ extract_landmarks.py).")
    parser.add_argument("--output_csv", type=str, default=cfg.normalized_csv_path,
                         help="[mode images] CSV landmark đã chuẩn hoá đầu ra.")

    # --- videos ---
    parser.add_argument("--input_dir", type=str, default=cfg.landmarks_seq_dir,
                         help="[mode videos] Thư mục .npy landmark thô đầu vào (từ extract_landmarks.py).")
    parser.add_argument("--output_dir", type=str, default=cfg.sequences_output_dir,
                         help="[mode videos] Thư mục lưu .npy đã chuẩn hoá + ghép chuỗi.")
    parser.add_argument("--labels_csv", type=str, default=cfg.sequences_labels_csv,
                         help="[mode videos] CSV ánh xạ [file_name, label, session_id].")

    # --- tham số chuẩn hoá (dùng chung) ---
    parser.add_argument("--origin_index", type=int, default=cfg.normalize.origin_index,
                         help="Chỉ số landmark làm gốc toạ độ (mặc định 0 = cổ tay).")
    parser.add_argument("--round_decimals", type=int, default=cfg.normalize.round_decimals,
                         help="Số chữ số thập phân làm tròn sau khi chuẩn hoá.")

    # --- tham số chuỗi (chỉ dùng khi --mode videos) ---
    parser.add_argument("--seq_len", type=int, default=cfg.sequence.seq_len,
                         help="[mode videos] Độ dài cố định (số frame) của mỗi mẫu chuỗi.")
    parser.add_argument("--padding", type=str, choices=["pre", "post"], default=cfg.sequence.padding,
                         help="[mode videos] Vị trí thêm frame 0 khi chuỗi ngắn hơn seq_len.")
    parser.add_argument("--truncating", type=str, choices=["pre", "post"], default=cfg.sequence.truncating,
                         help="[mode videos] Vị trí cắt bớt khi chuỗi dài hơn seq_len.")
    parser.add_argument("--use_sliding_window", action="store_true", default=cfg.sequence.use_sliding_window,
                         help="[mode videos] Bật sliding window để cắt 1 video dài thành nhiều mẫu.")
    parser.add_argument("--stride", type=int, default=cfg.sequence.stride,
                         help="[mode videos] Bước nhảy (frame) giữa 2 cửa sổ sliding window liên tiếp.")
    parser.add_argument("--missing_frame_strategy", type=str,
                         choices=["zero", "interpolate", "drop_sample"],
                         default=cfg.sequence.missing_frame_strategy,
                         help="[mode videos] Cách xử lý frame bị MediaPipe miss.")
    parser.add_argument("--max_missing_ratio", type=float, default=cfg.sequence.max_missing_ratio,
                         help="[mode videos] Tỉ lệ frame miss tối đa cho phép trước khi loại cả mẫu.")

    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = PreprocessConfig()
    cfg.normalize.origin_index = args.origin_index
    cfg.normalize.round_decimals = args.round_decimals

    if args.mode == "images":
        normalize_image_dataset(args.input_csv, args.output_csv, cfg)
    else:
        cfg.sequence.seq_len = args.seq_len
        cfg.sequence.padding = args.padding
        cfg.sequence.truncating = args.truncating
        cfg.sequence.use_sliding_window = args.use_sliding_window
        cfg.sequence.stride = args.stride
        cfg.sequence.missing_frame_strategy = args.missing_frame_strategy
        cfg.sequence.max_missing_ratio = args.max_missing_ratio
        normalize_video_dataset(args.input_dir, args.output_dir, args.labels_csv, cfg)


if __name__ == "__main__":
    main()
