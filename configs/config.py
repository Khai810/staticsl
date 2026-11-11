"""
configs/config.py
=================
Cấu hình tập trung cho toàn bộ pipeline nhận diện ngôn ngữ ký hiệu ASL:
- Tiền xử lý & chuẩn hóa Landmarks (Baseline 12-9, S-TR 14-9, và Optimized ST-Hybrid)
- Trích xuất MediaPipe Hand Landmarks (extract_landmarks.py)
- Augmentation nhẹ (Light Augmentation)
- Siêu tham số huấn luyện mô hình (Training Hyperparameters)
- Đường dẫn dữ liệu và checkpoints
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

# MediaPipe 21 Hand Landmarks Graph connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17)
]

FINGER_CHAINS = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'pinky': [0, 17, 18, 19, 20]
}

# 29 classes ASL chuẩn
ASL_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]


@dataclass
class MediaPipeConfig:
    """Tham số khởi tạo MediaPipe Hands — dùng cho extract_landmarks.py."""
    static_image_mode: bool = True
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    model_asset_path: str = "src/models/hand_landmarker.task"
    min_hand_presence_confidence: float = 0.5
    use_gpu_delegate: bool = False


@dataclass
class NormalizeConfig:
    """Tham số cho bước chuẩn hoá toạ độ landmark."""
    origin_index: int = 0
    reference_pair: Tuple[int, int] = (0, 9)
    normalize_z: bool = True
    round_decimals: int = 6


@dataclass
class SequenceConfig:
    seq_len: int = 30
    padding: str = "pre"
    truncating: str = "post"
    use_sliding_window: bool = True
    stride: int = 10
    missing_frame_strategy: str = "interpolate"
    max_missing_ratio: float = 0.3


@dataclass
class HybridConfig:
    image_size: int = 64
    rotation_range_deg: float = 20.0
    horizontal_flip_prob: float = 0.5
    gaussian_noise_std: float = 0.02
    brightness_delta: float = 0.2
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    random_crop_pad: int = 8
    use_mediapipe_crop: bool = True
    aux_bbox_padding_ratio: float = 0.20
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    adaptive_thresh_block_size: int = 11
    adaptive_thresh_C: int = 2
    cnn_filters: Tuple[int, int] = (32, 64)
    cnn_kernel_size: int = 3
    pool_size: int = 2
    patch_size: int = 16
    vit_num_layers: int = 2
    vit_num_heads: int = 4
    vit_embed_dim: int = 128
    vit_mlp_dim: int = 256
    vit_dropout: float = 0.1
    fc_hidden_dim: int = 128
    fc_dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    early_stopping_patience: int = 60
    lr_schedule: str = "cosine_decay"
    hybrid_class_labels: List[str] = field(
        default_factory=lambda: [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]
    )


@dataclass
class YoloConfig:
    bbox_padding_ratio: float = 0.15
    image_size: int = 224
    label_format: str = "yolo"


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify: bool = True
    group_column: str = "session_id"


@dataclass
class PreprocessConfig:
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Subconfigs
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    
    # Data paths & directories (for extract/normalize compatibility)
    raw_images_dir: str = os.path.join(project_root, "data/raw/asl_alphabet")
    raw_videos_dir: str = os.path.join(project_root, "data/raw/custom_captured_sequences")
    landmarks_csv_path: str = os.path.join(project_root, "data/landmarks/landmarks_raw.csv")
    raw_landmarks_csv: str = os.path.join(project_root, "data/landmarks/landmarks_raw.csv")
    landmarks_seq_dir: str = os.path.join(project_root, "data/landmarks/sequences_raw")
    normalized_csv_path: str = os.path.join(project_root, "data/landmarks/landmarks_normalized.csv")
    sequences_output_dir: str = os.path.join(project_root, "data/sequences")
    sequences_labels_csv: str = os.path.join(project_root, "data/sequences/sequence_labels.csv")
    yolo_dataset_dir: str = os.path.join(project_root, "data/yolo_dataset")
    
    # Manifest dir & paths for compatibility
    hybrid_manifest_dir: str = os.path.join(project_root, "data/hybrid_manifest")
    hybrid_train_manifest: str = os.path.join(project_root, "data/hybrid_manifest/train.csv")
    hybrid_val_manifest: str = os.path.join(project_root, "data/hybrid_manifest/val.csv")
    hybrid_test_manifest: str = os.path.join(project_root, "data/hybrid_manifest/test.csv")
    train_manifest: str = os.path.join(project_root, "data/hybrid_manifest/train.csv")
    val_manifest: str = os.path.join(project_root, "data/hybrid_manifest/val.csv")
    test_manifest: str = os.path.join(project_root, "data/hybrid_manifest/test.csv")
    
    # Checkpoints
    ckpt_12_9: str = os.path.join(project_root, "checkpoints/model2_hybrid_transformer_cnn_best_12-9.keras")
    ckpt_14_9: str = os.path.join(project_root, "checkpoints/model2_hybrid_transformer_cnn_best14-9.keras")
    ckpt_optimized: str = os.path.join(project_root, "checkpoints/model_optimized_st_hybrid.keras")
    hybrid_checkpoint_path: str = os.path.join(project_root, "checkpoints/model2_hybrid_transformer_cnn_best.keras")
    
    # Output paths
    plots_dir: str = os.path.join(project_root, "plots")
    reports_dir: str = os.path.join(project_root, "reports")
    
    # Labels
    class_labels: List[str] = field(default_factory=lambda: ASL_CLASSES)
    num_classes: int = len(ASL_CLASSES)
    
    # Model parameters
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Augmentation
    rotation_range_deg: float = 12.0
    scale_range: Tuple[float, float] = (0.93, 1.07)
    jitter_std: float = 0.008
    joint_dropout_prob: float = 0.03
