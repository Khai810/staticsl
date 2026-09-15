"""
src/models/model2_hybrid_transformer_cnn/dataset.py
=======================================================
Keras Sequence đọc manifest CSV và sinh ra batch tọa độ Landmarks
(thay vì ảnh thô).

    - inputs: shape (batch_size, 21, 3) (Toạ độ 21 khớp đã chuẩn hoá)
    - targets: shape (batch_size, num_classes) (one-hot labels)
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List
from configs.config import PreprocessConfig


class LandmarkHybridDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_path: str, cfg: PreprocessConfig, class_labels: list, training: bool, seed: int = 42):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.class_labels = class_labels
        self.label_to_id = {label: idx for idx, label in enumerate(class_labels)}
        self.training = training
        self.batch_size = cfg.hybrid.batch_size
        self.rng = np.random.default_rng(seed)

        # Lấy tên 63 cột tọa độ: r0 đến r62
        self.feature_cols = [f"r{i}" for i in range(63)]
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.training and len(self.df) > 0:
            self.df = self.df.sample(frac=1.0, random_state=int(self.rng.integers(0, 1000000))).reset_index(drop=True)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        origin = landmarks[self.cfg.normalize.origin_index]
        translated = landmarks - origin
        idx1, idx2 = self.cfg.normalize.reference_pair
        ref_distance = np.linalg.norm(translated[idx1] - translated[idx2])

        scaled = translated / ref_distance if ref_distance > 1e-6 else translated
        if not self.cfg.normalize.normalize_z:
            scaled[:, 2] = translated[:, 2]
        return scaled

    def __getitem__(self, idx: int):
        batch_df = self.df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Trích xuất toàn bộ 63 tính năng của batch và reshape về (Batch, 21, 3)
        flat_landmarks = batch_df[self.feature_cols].values.astype(np.float32)
        batch_landmarks = flat_landmarks.reshape(-1, 21, 3)

        X_batch, y_batch = [], []

        for i, (_, row) in enumerate(batch_df.iterrows()):
            landmarks = self._normalize_landmarks(batch_landmarks[i])

            if self.training:
                noise = self.rng.normal(0, self.cfg.hybrid.gaussian_noise_std, landmarks.shape)
                landmarks += noise

            X_batch.append(landmarks)
            y_batch.append(self.label_to_id[row["label"]])

        X_arr = np.stack(X_batch).astype(np.float32)
        y_arr = tf.keras.utils.to_categorical(y_batch, num_classes=len(self.class_labels))

        return X_arr, y_arr