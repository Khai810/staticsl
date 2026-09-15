"""
src/preprocessing/hybrid_transforms.py
=========================================
File RIÊNG cho Model 2 (Hybrid Transformer-CNN). Cài đặt lại đúng các bước
tiền xử lý mô tả trong bài báo tham khảo:

    Aly, M. & Fathi, I. S. "Recognizing American Sign Language gestures
    efficiently and accurately using a hybrid transformer model."
    Scientific Reports 15, 20253 (2025). DOI: 10.1038/s41598-025-06344-8

Khác với Model 1/2(cũ)/3, Model 2 này KHÔNG dùng landmark — toàn bộ xử lý
diễn ra TRỰC TIẾP trên pixel ảnh, sinh ra 2 ảnh riêng biệt cho 2 nhánh CNN:

    - PRIMARY  (ảnh toàn cục)     : ảnh gốc đã resize + augment, giữ nguyên
                                     bối cảnh xung quanh bàn tay.
    - AUXILIARY (ảnh tăng cường tay): vùng tay được CROP (nhờ bbox MediaPipe,
                                     tái dùng extract_landmarks.py), sau đó
                                     áp dụng adaptive threshold + Canny edge
                                     + histogram equalization để làm nổi bật
                                     đường viền/hình dạng tay, giảm nhiễu nền.

Hai ảnh này được đưa vào 2 nhánh CNN song song của model.py, rồi fusion
bằng phép nhân element-wise (xem model.py).
"""

from typing import Tuple, Optional

import cv2
import numpy as np

from configs.config import HybridConfig


# ============================================================================
# BƯỚC 1 — RESIZE + NORMALIZE (áp dụng cho CẢ train/val/test, không random)
# ============================================================================

def resize_and_normalize(image_bgr: np.ndarray, cfg: HybridConfig) -> np.ndarray:
    """
    Resize ảnh về (image_size, image_size) và chuẩn hoá pixel về [0, 1].

    Tham số:
        image_bgr (np.ndarray): ảnh gốc đọc bằng OpenCV (BGR), kích thước bất kỳ.
        cfg (HybridConfig):
            - image_size (int): kích thước cạnh vuông sau resize (mặc định 64,
              đúng theo bài báo — ảnh gốc Kaggle 200x200 được resize xuống 64x64
              để giảm tải tính toán, vẫn giữ tỉ lệ khung hình theo chiều vuông).

    Trả về:
        np.ndarray dtype float32, shape (image_size, image_size, 3), giá trị [0,1].
    """
    resized = cv2.resize(image_bgr, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


# ============================================================================
# BƯỚC 2 — DATA AUGMENTATION (CHỈ áp dụng cho tập TRAIN)
# ============================================================================

def augment_image(image_float: np.ndarray, cfg: HybridConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Áp dụng augmentation ngẫu nhiên lên 1 ảnh ĐÃ chuẩn hoá [0,1], theo đúng
    danh sách kỹ thuật augmentation mô tả trong bài báo:
        - Xoay ngẫu nhiên trong khoảng ±rotation_range_deg độ.
        - Lật ngang với xác suất horizontal_flip_prob.
        - Thêm nhiễu Gaussian (độ lệch chuẩn gaussian_noise_std).
        - Điều chỉnh độ sáng/tương phản ngẫu nhiên.
        - Random crop + pad để mô phỏng dịch chuyển vị trí tay trong khung hình.

    Tham số:
        image_float (np.ndarray): ảnh float32 [0,1], shape (H, W, 3).
        cfg (HybridConfig): xem giải thích từng tham số augmentation trong config.py.
        rng (np.random.Generator): bộ sinh số ngẫu nhiên (truyền vào để tái lập kết quả
                                    khi cần — dùng np.random.default_rng(seed)).

    Trả về:
        np.ndarray float32 [0,1], cùng shape với đầu vào.

    LƯU Ý: hàm này CHỈ được gọi cho tập TRAIN. Tập val/test giữ nguyên ảnh
    gốc (chỉ resize + normalize) để đánh giá công bằng, không augment.
    """
    h, w = image_float.shape[:2]
    img = image_float.copy()

    # --- Xoay ngẫu nhiên ±rotation_range_deg ---
    angle = rng.uniform(-cfg.rotation_range_deg, cfg.rotation_range_deg)
    rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # --- Lật ngang với xác suất 50% ---
    if rng.random() < cfg.horizontal_flip_prob:
        img = cv2.flip(img, 1)

    # --- Random crop + pad (mô phỏng dịch chuyển vị trí tay) ---
    pad = cfg.random_crop_pad
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    x_off = rng.integers(0, 2 * pad + 1)
    y_off = rng.integers(0, 2 * pad + 1)
    img = img_padded[y_off: y_off + h, x_off: x_off + w]

    # --- Điều chỉnh độ sáng ---
    brightness = rng.uniform(-cfg.brightness_delta, cfg.brightness_delta)
    img = np.clip(img + brightness, 0.0, 1.0)

    # --- Điều chỉnh tương phản ---
    contrast = rng.uniform(cfg.contrast_range[0], cfg.contrast_range[1])
    mean = img.mean()
    img = np.clip((img - mean) * contrast + mean, 0.0, 1.0)

    # --- Nhiễu Gaussian ---
    noise = rng.normal(0.0, cfg.gaussian_noise_std, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)

    return img.astype(np.float32)


# ============================================================================
# BƯỚC 3 — TẠO ẢNH NHÁNH AUXILIARY (tăng cường vùng tay, giảm nhiễu nền)
# ============================================================================

def crop_hand_region(
    image_bgr: np.ndarray, bbox_xyxy: Optional[list], cfg: HybridConfig
) -> np.ndarray:
    """
    Cắt vùng tay từ ảnh gốc dựa theo bbox lấy từ MediaPipe (mediapipe_extractor
    / extract_landmarks.py). Nếu không có bbox (MediaPipe không phát hiện
    được tay), trả về NGUYÊN ảnh gốc để nhánh auxiliary vẫn có input hợp lệ.

    Tham số:
        image_bgr (np.ndarray): ảnh gốc (chưa resize).
        bbox_xyxy (Optional[list]): [x_min, y_min, x_max, y_max] pixel, lấy
            từ HandLandmarkExtractor.extract(...).bbox_xyxy, hoặc None.
        cfg (HybridConfig):
            - aux_bbox_padding_ratio (float): tỉ lệ nới rộng bbox mỗi phía
              trước khi crop, tránh cắt sát mất đầu ngón tay.

    Trả về:
        np.ndarray — vùng ảnh đã crop (hoặc ảnh gốc nếu bbox=None).
    """
    if bbox_xyxy is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = bbox_xyxy
    box_w, box_h = x_max - x_min, y_max - y_min
    pad_w, pad_h = box_w * cfg.aux_bbox_padding_ratio, box_h * cfg.aux_bbox_padding_ratio

    x_min = max(0, int(x_min - pad_w))
    y_min = max(0, int(y_min - pad_h))
    x_max = min(w, int(x_max + pad_w))
    y_max = min(h, int(y_max + pad_h))

    if x_max <= x_min or y_max <= y_min:
        return image_bgr  # bbox không hợp lệ -> fallback ảnh gốc

    return image_bgr[y_min:y_max, x_min:x_max]


def enhance_hand_region(image_bgr: np.ndarray, cfg: HybridConfig) -> np.ndarray:
    """
    Tăng cường vùng tay bằng 3 kỹ thuật xử lý ảnh cổ điển, ĐÚNG theo mô tả
    trong bài báo (mục "Hand region enhancement and background noise
    reduction"):
        1. Adaptive thresholding — tách tay khỏi nền.
        2. Canny edge detection — làm nổi bật đường viền bàn tay.
        3. Histogram equalization — chuẩn hoá độ sáng, giảm ảnh hưởng ánh sáng.

    Tham số:
        image_bgr (np.ndarray): ảnh (thường là vùng tay đã crop_hand_region).
        cfg (HybridConfig):
            - adaptive_thresh_block_size (int): kích thước block (số lẻ) cho
              adaptive threshold.
            - adaptive_thresh_C (int): hằng số trừ đi trong công thức adaptive threshold.
            - canny_threshold1, canny_threshold2 (int): 2 ngưỡng cho Canny edge detector.

    Trả về:
        np.ndarray dtype uint8, shape (H, W, 3) — ảnh RGB đã tăng cường
        (kết hợp: kênh đã cân bằng histogram + đường biên Canny chồng lên).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Histogram equalization — chuẩn hoá độ sáng.
    equalized = cv2.equalizeHist(gray)

    # 2. Adaptive threshold — tách tay khỏi nền theo từng vùng cục bộ.
    block_size = cfg.adaptive_thresh_block_size
    if block_size % 2 == 0:
        block_size += 1  # bắt buộc số lẻ theo yêu cầu của OpenCV
    thresholded = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, cfg.adaptive_thresh_C,
    )

    # 3. Canny edge detection — làm nổi bật đường viền tay.
    edges = cv2.Canny(equalized, cfg.canny_threshold1, cfg.canny_threshold2)

    # Kết hợp: dùng threshold làm mask, chồng biên Canny để giữ chi tiết viền tay.
    combined = cv2.bitwise_and(equalized, equalized, mask=thresholded)
    combined = cv2.bitwise_or(combined, edges)

    enhanced_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb


def build_auxiliary_image(
    image_bgr: np.ndarray, bbox_xyxy: Optional[list], cfg: HybridConfig
) -> np.ndarray:
    """
    HÀM TỔNG HỢP cho nhánh Auxiliary: crop vùng tay -> tăng cường -> resize
    + normalize về đúng kích thước chuẩn (image_size, image_size, 3), [0,1].

    Tham số:
        image_bgr (np.ndarray): ảnh gốc (BGR, chưa resize).
        bbox_xyxy (Optional[list]): bbox tay lấy từ MediaPipe, hoặc None.
        cfg (HybridConfig): xem crop_hand_region() và enhance_hand_region().

    Trả về:
        np.ndarray float32 [0,1], shape (image_size, image_size, 3).
    """
    cropped = crop_hand_region(image_bgr, bbox_xyxy, cfg)
    enhanced = enhance_hand_region(cropped, cfg)
    return resize_and_normalize(enhanced, cfg)


def build_primary_image(image_bgr: np.ndarray, cfg: HybridConfig) -> np.ndarray:
    """
    HÀM TỔNG HỢP cho nhánh Primary: chỉ resize + normalize ảnh GỐC (giữ
    nguyên bối cảnh toàn cục, KHÔNG tăng cường/crop).

    Trả về:
        np.ndarray float32 [0,1], shape (image_size, image_size, 3).
    """
    return resize_and_normalize(image_bgr, cfg)


def build_dual_path_pair(
    image_bgr: np.ndarray,
    bbox_xyxy: Optional[list],
    cfg: HybridConfig,
    training: bool,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HÀM TỔNG HỢP CAO NHẤT — sinh ra CẶP ẢNH (primary, auxiliary) sẵn sàng
    đưa vào model, dùng chung cho cả lúc huấn luyện (có augment) và lúc
    inference/đánh giá (không augment).

    Tham số:
        image_bgr (np.ndarray): ảnh gốc đọc từ đĩa (BGR), kích thước bất kỳ.
        bbox_xyxy (Optional[list]): bbox tay lấy từ MediaPipe cho ảnh này
            (hoặc None nếu không dùng MediaPipe / không detect được).
        cfg (HybridConfig): cấu hình đầy đủ.
        training (bool): True -> áp dụng augment_image() cho CẢ 2 nhánh
                          (đồng bộ phép biến đổi hình học để 2 nhánh vẫn
                          khớp không gian với nhau); False -> chỉ resize/normalize.
        rng (np.random.Generator): bắt buộc truyền vào nếu training=True.

    Trả về:
        (primary_image, auxiliary_image): 2 np.ndarray float32 [0,1],
        shape (image_size, image_size, 3) mỗi ảnh.
    """
    primary = build_primary_image(image_bgr, cfg)
    auxiliary = build_auxiliary_image(image_bgr, bbox_xyxy, cfg)

    if training:
        assert rng is not None, "Phải truyền rng khi training=True để augment có thể tái lập."
        # Dùng CÙNG 1 rng nhưng gọi augment RIÊNG cho từng nhánh — chấp nhận
        # augment hơi lệch nhau giữa 2 nhánh (giúp mô hình bền hơn với sai
        # lệch crop nhỏ), đúng tinh thần augmentation độc lập của bài báo.
        primary = augment_image(primary, cfg, rng)
        auxiliary = augment_image(auxiliary, cfg, rng)

    return primary, auxiliary
