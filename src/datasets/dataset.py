"""
src/datasets/dataset.py
=======================
Dataset module for loading landmark CSVs with flexible normalization and augmentation modes.
Supports:
- Norm modes: '12_9' (Wrist centered + middle MCP scale, keeps orientation),
              '14_9' (Wrist centered + destructive rotation canonicalization),
              'optimized' (Robust palm scale + multi-stream features)
- Light physiological skeleton data augmentation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional

try:
    from src.preprocessing.normalization import normalize_12_9, normalize_14_9, normalize_robust_optimized
    from src.preprocessing.augmentation import augment_landmarks_light
except ImportError:
    from src.normalization import normalize_12_9, normalize_14_9, normalize_robust_optimized
    from src.augmentation import augment_landmarks_light


class FlexibleLandmarkDataset(tf.keras.utils.Sequence):
    """
    Keras Sequence providing landmark batches with configurable normalization:
    - norm_mode: '12_9', '14_9', or 'optimized'
    - augment: bool, whether to apply light skeleton augmentation
    """
    def __init__(
        self,
        csv_path: str,
        class_labels: List[str],
        batch_size: int = 64,
        norm_mode: str = "12_9",
        augment: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.class_labels = class_labels
        self.label_to_id = {label: idx for idx, label in enumerate(class_labels)}
        self.batch_size = batch_size
        self.norm_mode = norm_mode
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        # 63 landmark coordinates: r0 to r62
        self.feature_cols = [f"r{i}" for i in range(63)]
        self.has_handedness = "handedness" in self.df.columns
        
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle and len(self.df) > 0:
            self.df = self.df.sample(frac=1.0, random_state=int(self.rng.integers(0, 1000000))).reset_index(drop=True)

    def _process_sample(self, raw_lm: np.ndarray, handedness: Optional[str] = None) -> np.ndarray:
        if self.norm_mode == "12_9":
            norm_lm = normalize_12_9(raw_lm)
        elif self.norm_mode == "14_9":
            norm_lm = normalize_14_9(raw_lm, handedness=handedness)
        elif self.norm_mode == "optimized":
            res = normalize_robust_optimized(raw_lm, handedness=handedness)
            norm_lm = res['joints']
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        if self.augment:
            norm_lm = augment_landmarks_light(norm_lm, rng=self.rng)
            
        return norm_lm

    def __getitem__(self, idx: int):
        batch_df = self.df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        flat_landmarks = batch_df[self.feature_cols].values.astype(np.float32)
        batch_landmarks = flat_landmarks.reshape(-1, 21, 3)
        
        X_batch, y_batch = [], []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            handedness = row["handedness"] if self.has_handedness else None
            lm = self._process_sample(batch_landmarks[i], handedness)
            X_batch.append(lm)
            y_batch.append(self.label_to_id[row["label"]])
            
        X_arr = np.stack(X_batch).astype(np.float32)
        y_arr = tf.keras.utils.to_categorical(y_batch, num_classes=len(self.class_labels))
        return X_arr, y_arr
