"""
src/preprocessing/extract_landmarks.py
=========================================
File DÙNG CHUNG cho cả 3 model: ảnh/video -> 21 landmark -> vector 63D.

MediaPipe: 0.10.30+
API: MediaPipe Tasks -> HandLandmarker

Lưu ý:
    MediaPipe Tasks không dùng mp.solutions.hands.Hands() nữa.
    HandLandmarker yêu cầu một model asset dạng .task.

Model asset mặc định:
    models/hand_landmarker.task

Có thể cấu hình đường dẫn model bằng một trong các thuộc tính sau của
MediaPipeConfig:
    - model_asset_path
    - hand_landmarker_model_path

Vai trò cho từng model:
    - Model 1 (MLP)       : landmark từ ẢNH TĨNH (ASL Alphabet).
    - Model 2 (CNN+BiLSTM): landmark từ TỪNG FRAME VIDEO, giữ thứ tự thời gian.
    - Model 3 (YOLOv11n)  : landmark pixel -> bounding box.

CLI ảnh:
    python -m src.preprocessing.extract_landmarks \
        --mode images \
        --input_dir data/raw/asl_alphabet \
        --output_csv data/landmarks/landmarks_raw.csv

CLI video:
    python -m src.preprocessing.extract_landmarks \
        --mode videos \
        --input_dir data/raw/custom_captured_sequences \
        --output_dir data/landmarks/sequences_raw
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.config import PreprocessConfig, MediaPipeConfig


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NUM_LANDMARKS = 21
NUM_COORDS = 3
FLAT_VECTOR_SIZE = NUM_LANDMARKS * NUM_COORDS
DEFAULT_MODEL_ASSET_PATH = "src/models/hand_landmarker.task"


@dataclass
class ExtractionResult:
    """Kết quả trả về sau khi trích xuất 1 ảnh/frame."""

    success: bool
    landmarks_raw: Optional[np.ndarray]      # shape (21, 3)
    landmarks_flat: Optional[np.ndarray]     # shape (63,)
    bbox_xyxy: Optional[List[float]]         # [x_min, y_min, x_max, y_max] (pixel)


class HandLandmarkExtractor:
    """
    Wrapper cho MediaPipe Tasks HandLandmarker.

    Giữ nguyên interface cũ của pipeline:
        extractor = HandLandmarkExtractor(cfg)
        result = extractor.extract(image_bgr)

    Khác biệt nội bộ:
        - MediaPipe 0.10.30 dùng mediapipe.tasks.vision.HandLandmarker.
        - Ảnh tĩnh dùng RunningMode.IMAGE + detect().
        - Video dùng RunningMode.VIDEO + detect_for_video(timestamp_ms).
        - MediaPipe Tasks yêu cầu file model .task.
    """

    def __init__(self, cfg: MediaPipeConfig):
        self.cfg = cfg

        self._model_asset_path = self._get_model_asset_path(cfg)
        self._num_hands = int(getattr(cfg, "max_num_hands", 1))
        self._min_detection_confidence = float(
            getattr(cfg, "min_detection_confidence", 0.5)
        )
        self._min_tracking_confidence = float(
            getattr(cfg, "min_tracking_confidence", 0.5)
        )
        self._min_hand_presence_confidence = float(
            getattr(cfg, "min_hand_presence_confidence", 0.5)
        )
        self._static_image_mode = bool(getattr(cfg, "static_image_mode", True))

        self._timestamp_ms = 0
        self._landmarker = self._create_landmarker()

    @staticmethod
    def _get_model_asset_path(cfg: MediaPipeConfig) -> str:
        """Lấy đường dẫn model mà không bắt buộc sửa MediaPipeConfig cũ."""
        model_path = getattr(cfg, "model_asset_path", None)

        if model_path is None:
            model_path = getattr(cfg, "hand_landmarker_model_path", None)

        if model_path is None:
            model_path = DEFAULT_MODEL_ASSET_PATH

        model_path = str(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "Không tìm thấy Hand Landmarker model (.task): "
                f"{model_path}\n"
                "Hãy đặt model vào models/hand_landmarker.task hoặc "
                "cấu hình MediaPipeConfig.model_asset_path."
            )

        return model_path

    def _create_landmarker(self):
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if self._static_image_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._model_asset_path,
            # Bật ép buộc dùng GPU tại đây
            delegate=mp.tasks.BaseOptions.Delegate.GPU
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_hand_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _to_mp_image(image_bgr: np.ndarray) -> mp.Image:
        """OpenCV BGR -> MediaPipe Image RGB."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    def extract(
        self,
        image_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Trích xuất landmark từ 1 ảnh/frame.

        Với ảnh tĩnh: dùng detect().
        Với video: dùng detect_for_video() và timestamp tăng dần.
        """
        if image_bgr is None or image_bgr.ndim != 3:
            return ExtractionResult(False, None, None, None)

        mp_image = self._to_mp_image(image_bgr)

        if self._static_image_mode:
            result = self._landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                timestamp_ms = self._timestamp_ms

            result = self._landmarker.detect_for_video(
                mp_image,
                int(timestamp_ms),
            )
            self._timestamp_ms = int(timestamp_ms) + 1

        if not result.hand_landmarks:
            return ExtractionResult(False, None, None, None)

        # max_num_hands=1 trong pipeline hiện tại -> lấy bàn tay đầu tiên.
        hand = result.hand_landmarks[0]
        h, w = image_bgr.shape[:2]

        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32,
        )

        flat = raw.flatten()

        # Bbox pixel thực tế cho Model 3.
        xs_px = raw[:, 0] * w
        ys_px = raw[:, 1] * h
        bbox = [
            float(xs_px.min()),
            float(ys_px.min()),
            float(xs_px.max()),
            float(ys_px.max()),
        ]

        return ExtractionResult(True, raw, flat, bbox)

    def close(self):
        """Giải phóng tài nguyên MediaPipe Tasks."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# --------------------------------------------------------------------------- #
# CLI: ảnh tĩnh - Model 1
# --------------------------------------------------------------------------- #

def extract_from_image_dataset(
    input_dir: str,
    output_csv: str,
    cfg: PreprocessConfig,
) -> pd.DataFrame:
    """Trích xuất landmark thô từ toàn bộ dataset ảnh."""
    cfg.mediapipe.static_image_mode = True

    input_root = Path(input_dir)
    label_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())

    if not label_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục con (label) nào trong {input_root}. "
            f"Cấu trúc mong đợi: {input_root}/<label>/<anh>.jpg"
        )

    rows = []
    n_missed = 0
    n_total = 0

    with HandLandmarkExtractor(cfg.mediapipe) as extractor:
        for label_dir in label_dirs:
            label = label_dir.name
            image_files = sorted(
                f
                for f in label_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

            for img_path in tqdm(
                image_files,
                desc=f"[extract_landmarks] Label {label}",
            ):
                n_total += 1
                image_bgr = cv2.imread(str(img_path))

                if image_bgr is None:
                    n_missed += 1
                    continue

                result = extractor.extract(image_bgr)
                if not result.success:
                    n_missed += 1
                    continue

                row = {
                    f"r{i}": value
                    for i, value in enumerate(result.landmarks_flat)
                }
                row["label"] = label
                row["source_file"] = img_path.name
                rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    miss_ratio = n_missed / n_total if n_total else 0.0
    print(f"\n[extract_landmarks] Tổng số ảnh xử lý: {n_total}")
    print(
        "[extract_landmarks] Số ảnh MediaPipe KHÔNG phát hiện được tay: "
        f"{n_missed} ({miss_ratio:.2%})"
    )
    print(
        "[extract_landmarks] -> Ghi số liệu này vào báo cáo "
        "(mục hạn chế của phương pháp)."
    )
    print(f"[extract_landmarks] Đã lưu: {output_csv} — shape: {df.shape}")

    return df


# --------------------------------------------------------------------------- #
# CLI: video - Model 2
# --------------------------------------------------------------------------- #

def extract_from_video_dataset(
    input_dir: str,
    output_dir: str,
    cfg: PreprocessConfig,
) -> None:
    """
    Trích xuất landmark thô theo từng frame của video.

    Output mỗi video:
        shape (T, 63)

    Frame không phát hiện được tay -> 63 giá trị NaN.
    """
    cfg.mediapipe.static_image_mode = False

    input_root = Path(input_dir)
    label_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())

    if not label_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục con (label) nào trong {input_root}. "
            f"Cấu trúc mong đợi: {input_root}/<label>/<video>.mp4"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with HandLandmarkExtractor(cfg.mediapipe) as extractor:
        for label_dir in label_dirs:
            label = label_dir.name
            video_files = sorted(
                f
                for f in label_dir.iterdir()
                if f.suffix.lower() in (".mp4", ".avi", ".mov")
            )

            for video_path in tqdm(
                video_files,
                desc=f"[extract_landmarks] Label {label}",
            ):
                session_id = video_path.stem
                frames = []
                cap = cv2.VideoCapture(str(video_path))
                frame_index = 0

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    # VIDEO mode của MediaPipe Tasks yêu cầu timestamp tăng dần.
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        timestamp_ms = int(frame_index * 1000 / fps)
                    else:
                        timestamp_ms = frame_index

                    result = extractor.extract(
                        frame_bgr,
                        timestamp_ms=timestamp_ms,
                    )

                    if result.success:
                        frames.append(result.landmarks_flat)
                    else:
                        frames.append(
                            np.full(
                                FLAT_VECTOR_SIZE,
                                np.nan,
                                dtype=np.float32,
                            )
                        )

                    frame_index += 1

                cap.release()

                if not frames:
                    print(f"[CẢNH BÁO] Video rỗng hoặc lỗi đọc: {video_path}")
                    continue

                sequence_raw = np.stack(frames)
                out_name = f"{label}__{session_id}.npy"
                np.save(output_path / out_name, sequence_raw)

    print(
        "[extract_landmarks] Đã lưu toàn bộ landmark thô (video) vào: "
        f"{output_dir}"
    )


def _parse_args() -> argparse.Namespace:
    cfg = PreprocessConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Trích xuất landmark thô bằng MediaPipe Tasks "
            "(dùng chung cho Model 1, 2, 3)."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["images", "videos"],
        required=True,
        help="'images' cho ảnh tĩnh; 'videos' cho video.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Thư mục dữ liệu: <input_dir>/<label>/<file>.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=cfg.landmarks_csv_path,
        help="[images] Đường dẫn CSV đầu ra.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg.landmarks_seq_dir,
        help="[videos] Thư mục lưu file .npy.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=cfg.mediapipe.min_detection_confidence,
        help="Ngưỡng tin cậy phát hiện tay (0.0 - 1.0).",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=cfg.mediapipe.min_tracking_confidence,
        help="Ngưỡng tin cậy tracking cho video (0.0 - 1.0).",
    )
    parser.add_argument(
        "--min_hand_presence_confidence",
        type=float,
        default=getattr(cfg.mediapipe, "min_hand_presence_confidence", 0.5),
        help="Ngưỡng tin cậy hand presence (0.0 - 1.0).",
    )
    parser.add_argument(
        "--model_asset_path",
        type=str,
        default=getattr(
            cfg.mediapipe,
            "model_asset_path",
            getattr(
                cfg.mediapipe,
                "hand_landmarker_model_path",
                DEFAULT_MODEL_ASSET_PATH,
            ),
        ),
        help="Đường dẫn Hand Landmarker .task.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = PreprocessConfig()

    cfg.mediapipe.min_detection_confidence = args.min_detection_confidence
    cfg.mediapipe.min_tracking_confidence = args.min_tracking_confidence
    cfg.mediapipe.min_hand_presence_confidence = args.min_hand_presence_confidence
    cfg.mediapipe.model_asset_path = args.model_asset_path

    if args.mode == "images":
        extract_from_image_dataset(
            args.input_dir,
            args.output_csv,
            cfg,
        )
    else:
        extract_from_video_dataset(
            args.input_dir,
            args.output_dir,
            cfg,
        )


if __name__ == "__main__":
    main()
