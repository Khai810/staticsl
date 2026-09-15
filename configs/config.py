"""
configs/config.py
====================
File cấu hình TẤT CẢ tham số dùng cho bước tiền xử lý (preprocessing).
Dùng chung cho Model 1 (MLP), Model 2 (1D-CNN + BiLSTM), Model 3 (YOLOv11n).

Ghi chú: roadmap gốc đặt tên "configs/config.yaml", ở đây dùng file .py với
dataclass thay vì .yaml để (1) có gợi ý kiểu dữ liệu (type hint) khi code,
(2) validate giá trị ngay lúc import. Nếu muốn dùng đúng định dạng YAML,
có thể thêm hàm load_from_yaml() ở cuối file này (đã để sẵn khung).

Cách dùng trong các script khác:
    from configs.config import PreprocessConfig
    cfg = PreprocessConfig()            # dùng giá trị mặc định
    cfg = PreprocessConfig(seq_len=45)  # ghi đè khi cần

Mọi script trong src/preprocessing/ đều import từ đây để đảm bảo TOÀN BỘ
pipeline (Model 1, 2, 3) dùng chung một bộ tham số chuẩn hoá.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class MediaPipeConfig:
    """Tham số khởi tạo MediaPipe Hands — dùng chung cho cả 3 model."""

    # static_image_mode:
    #   True  -> dùng cho ẢNH TĨNH rời rạc (Model 1, Model 3 khi build dataset)
    #   False -> dùng cho VIDEO/WEBCAM liên tục (Model 2, bật tracking giúp
    #            mượt và nhanh hơn giữa các frame liên tiếp)
    static_image_mode: bool = True

    # Số bàn tay tối đa cần phát hiện trong 1 khung hình.
    # Đề tài chỉ nhận diện 1 tay -> giữ = 1 để tăng tốc độ xử lý.
    max_num_hands: int = 1

    # Ngưỡng tin cậy tối thiểu để MediaPipe xác nhận có bàn tay trong ảnh.
    # Giá trị càng cao -> ít false positive nhưng dễ bỏ sót tay ở góc xấu.
    min_detection_confidence: float = 0.7

    # Ngưỡng tin cậy tối thiểu để tiếp tục TRACK bàn tay giữa các frame
    # (chỉ có tác dụng khi static_image_mode=False, tức video).
    min_tracking_confidence: float = 0.5

    # Độ phức tạp mô hình MediaPipe: 0 (nhanh, kém chính xác) hoặc 1 (chính
    # xác hơn, chậm hơn). Máy CPU yếu nên để 0; máy khá hơn nên để 1.
    model_complexity: int = 1
    model_asset_path: str = "src/models/hand_landmarker.task"


@dataclass
class NormalizeConfig:
    """Tham số cho bước chuẩn hoá toạ độ landmark — dùng chung cho cả 3 model."""

    # Chỉ số landmark làm GỐC TOẠ ĐỘ (origin). MediaPipe quy ước landmark 0
    # là cổ tay (wrist) -> mọi điểm khác trừ đi điểm này để bất biến VỊ TRÍ.
    origin_index: int = 0

    # Cặp landmark dùng để tính khoảng cách THAM CHIẾU, dùng để chia (scale)
    # toàn bộ toạ độ -> bất biến KHOẢNG CÁCH tay-camera.
    # (0, 9) = khoảng cách từ cổ tay đến gốc ngón giữa (giữa lòng bàn tay).
    reference_pair: Tuple[int, int] = (0, 9)

    # Có chuẩn hoá trục Z (độ sâu) hay không.
    normalize_z: bool = True

    # Số chữ số thập phân làm tròn sau khi chuẩn hoá.
    round_decimals: int = 6


@dataclass
class SequenceConfig:
    """
    [KHÔNG CÒN DÙNG] Tham số này thuộc phiên bản Model 2 CŨ (1D-CNN+BiLSTM
    trên chuỗi landmark). Từ khi đổi Model 2 sang Hybrid Transformer-CNN
    (dựa trên ảnh tĩnh, không dùng chuỗi thời gian), class này không còn
    được các script hiện tại sử dụng. Giữ lại trong config để không phá vỡ
    code cũ / dữ liệu chuỗi đã thu thập trước đó nếu bạn muốn dùng lại sau này.
    """

    # Độ dài cố định (số khung hình) của MỖI chuỗi đưa vào model.
    # Ví dụ quay ở 30 FPS, seq_len=30 nghĩa là mỗi mẫu ~1 giây chuyển động.
    seq_len: int = 30

    # Chiến lược PADDING khi chuỗi thu được NGẮN hơn seq_len:
    #   "pre"  -> thêm khung hình 0 vào ĐẦU chuỗi
    #   "post" -> thêm khung hình 0 vào CUỐI chuỗi
    padding: str = "pre"

    # Chiến lược CẮT BỚT khi chuỗi thu được DÀI hơn seq_len:
    #   "pre"  -> cắt bỏ phần ĐẦU, giữ lại đoạn cuối
    #   "post" -> cắt bỏ phần CUỐI, giữ lại đoạn đầu
    truncating: str = "post"

    # Có dùng SLIDING WINDOW để cắt 1 video dài thành nhiều mẫu ngắn hay không.
    use_sliding_window: bool = True

    # Bước nhảy (số frame) giữa 2 cửa sổ liên tiếp khi use_sliding_window=True.
    stride: int = 10

    # Cách xử lý khi 1 frame trong chuỗi bị MediaPipe MISS (không detect được tay):
    #   "zero"        -> điền vector 0 vào vị trí đó
    #   "interpolate" -> nội suy tuyến tính từ frame trước/sau gần nhất
    #   "drop_sample" -> bỏ luôn toàn bộ mẫu chuỗi nếu có bất kỳ frame miss
    missing_frame_strategy: str = "interpolate"

    # Tỉ lệ frame miss tối đa cho phép trong 1 chuỗi trước khi bị loại bỏ.
    max_missing_ratio: float = 0.3


@dataclass
class HybridConfig:
    """Tham số riêng cho Model 2 — Landmark-based Hybrid Transformer-CNN."""

    # Augmentation: Giờ đây ta dùng toạ độ nên sẽ áp dụng "Jittering" toạ độ
    gaussian_noise_std: float = 0.02  # Biên độ nhiễu cộng vào điểm khớp

    # --- Kiến trúc Nhánh 1: 1D-CNN ---
    cnn_filters: Tuple[int, int] = (32, 64)

    # --- Kiến trúc Nhánh 2: Spatial Transformer ---
    vit_num_layers: int = 2       # 2 lớp encoder
    vit_num_heads: int = 4        # 4 attention heads
    vit_embed_dim: int = 128      # Kích thước embedding của mỗi token (điểm khớp)
    vit_mlp_dim: int = 256        # FFN hidden layer
    vit_dropout: float = 0.1

    # --- Classification head ---
    fc_hidden_dim: int = 128
    fc_dropout: float = 0.3

    # --- Huấn luyện ---
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    early_stopping_patience: int = 10
    lr_schedule: str = "cosine_decay"

    hybrid_class_labels: List[str] = field(
        default_factory=lambda: [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + ["del", "nothing", "space"]
    )


@dataclass
class YoloConfig:
    """Tham số riêng cho Model 3 (YOLOv11n) — sinh bounding box + format YOLO."""

    # Tỉ lệ nới rộng (padding) bbox quanh landmark, giúp box không bó sát tay
    # (đơn vị: % chiều rộng/cao bbox gốc). 0.15 = nới thêm 15% mỗi phía.
    bbox_padding_ratio: float = 0.15

    # Kích thước ảnh output chuẩn hoá (vuông) trước khi đưa vào YOLO.
    image_size: int = 224

    # Định dạng nhãn YOLO: "class_id x_center y_center width height" (đã
    # chuẩn hoá 0..1 theo kích thước ảnh) — giữ mặc định chuẩn Ultralytics.
    label_format: str = "yolo"


@dataclass
class SplitConfig:
    """Tham số chia tập dữ liệu — dùng chung cho cả 3 model."""

    test_size: float = 0.15       # tỉ lệ tập test
    val_size: float = 0.15        # tỉ lệ tập validation (tính trên toàn bộ dataset)
    random_state: int = 42        # seed để tái lập kết quả chia tập
    stratify: bool = True         # chia đều tỉ lệ theo từng lớp (class)
    # Cột dùng để GOM NHÓM khi chia tập (áp dụng cho Model 2), đảm bảo các
    # chuỗi cắt từ CÙNG 1 video/phiên quay không bị chia vào cả train và
    # test (tránh rò rỉ dữ liệu - data leakage).
    group_column: str = "session_id"


@dataclass
class PreprocessConfig:
    """Gộp toàn bộ cấu hình con lại làm 1 điểm import duy nhất."""

    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    # Danh sách nhãn ký tự (A-Z). Sửa lại nếu đề tài chỉ dùng 24 ký tự
    # (bỏ J, Z vì cần chuyển động — tuỳ phạm vi Model 1 tĩnh thuần).
    class_labels: List[str] = field(
        default_factory=lambda: [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    )

    # Các ký tự có YẾU TỐ CHUYỂN ĐỘNG trong ASL thực tế (J, Z). Model 1 (MLP
    # trên landmark tĩnh) và Model 3 (YOLO) nên loại các ký tự này khỏi tập
    # huấn luyện nếu chỉ thu ảnh tĩnh. Model 2 (Hybrid Transformer-CNN) thì
    # KHÔNG bị giới hạn bởi danh sách này, vì dùng bộ ảnh Kaggle ASL Alphabet
    # (ảnh J/Z trong bộ này vẫn là tư thế tĩnh, không phải video).
    dynamic_labels: List[str] = field(default_factory=lambda: ["J", "Z"])

    # ---- Đường dẫn dữ liệu (chỉnh lại theo máy của bạn) ----
    raw_images_dir: str = "data/raw/asl_alphabet/asl_alphabet_train"
    raw_videos_dir: str = "data/raw/custom_captured_sequences"  # KHÔNG CÒN DÙNG cho Model 2 (đã bỏ BiLSTM)
    landmarks_csv_path: str = "data/landmarks/landmarks_raw.csv"        # output extract_landmarks.py (Model 1)
    landmarks_seq_dir: str = "data/landmarks/sequences_raw"             # (giữ lại, không dùng cho Model 2 nữa)
    normalized_csv_path: str = "data/landmarks/landmarks_normalized.csv"  # output normalize_landmarks.py (Model 1)
    sequences_output_dir: str = "data/sequences"                        # (giữ lại, không dùng cho Model 2 nữa)
    sequences_labels_csv: str = "data/sequences/sequence_labels.csv"
    yolo_dataset_dir: str = "data/yolo_dataset"                         # output convert_to_yolo_format.py (Model 3)

    # ---- Đường dẫn RIÊNG cho Model 2 (Hybrid Transformer-CNN) ----
    # Model 2 dùng TRỰC TIẾP bộ ảnh gốc trong raw_images_dir (ASL Alphabet,
    # 29 lớp) — không cần bước trích xuất landmark. prepare_hybrid_dataset.py
    # chỉ tạo ra 3 file manifest CSV (đường dẫn ảnh + nhãn) cho train/val/test,
    # việc resize/augment/tách dual-path được làm ON-THE-FLY lúc huấn luyện.
    hybrid_manifest_dir: str = "data/hybrid_manifest"
    hybrid_train_manifest: str = "data/hybrid_manifest/train.csv"
    hybrid_val_manifest: str = "data/hybrid_manifest/val.csv"
    hybrid_test_manifest: str = "data/hybrid_manifest/test.csv"
    hybrid_checkpoint_path: str = "checkpoints/model2_hybrid_transformer_cnn_best.keras"
