"""
src/training/train_optimized.py
===============================
Script huấn luyện mô hình Optimized ST-Hybrid Model:
- Áp dụng Light Skeleton Augmentation (rotation, scaling, jitter, dropout)
- Sử dụng Cosine Decay Learning Rate với AdamW và Label Smoothing
- Tự động lưu checkpoint tốt nhất vào checkpoints/model_optimized_st_hybrid.keras
- Xuất lịch sử huấn luyện ra reports/optimized_training_history.json
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Hỗ trợ chạy trực tiếp từ bất kỳ thư mục nào
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import PreprocessConfig
from src.models.st_hybrid import build_optimized_st_hybrid_model
from src.datasets.dataset import FlexibleLandmarkDataset


def train_optimized_model(epochs: int = 20, batch_size: int = 64):
    cfg = PreprocessConfig()
    print("=" * 60)
    print("KHỞI TẠO HUẤN LUYỆN: OPTIMIZED ST-HYBRID MODEL")
    print("=" * 60)
    print(f"Project root: {cfg.project_root}")
    print(f"Train manifest: {cfg.train_manifest}")
    print(f"Val manifest:   {cfg.val_manifest}")
    
    print("\nKhởi tạo Datasets...")
    train_ds = FlexibleLandmarkDataset(
        csv_path=cfg.train_manifest,
        class_labels=cfg.class_labels,
        batch_size=batch_size,
        norm_mode="12_9", # Giữ nguyên định hướng toàn cục (không canonicalize quay)
        augment=True,     # Bật Light Skeleton Augmentation
        shuffle=True,
        seed=42
    )
    
    val_ds = FlexibleLandmarkDataset(
        csv_path=cfg.val_manifest,
        class_labels=cfg.class_labels,
        batch_size=batch_size,
        norm_mode="12_9",
        augment=False,
        shuffle=False
    )
    
    print(f"\nKhởi tạo kiến trúc mô hình ({cfg.num_classes} classes)...")
    model = build_optimized_st_hybrid_model(
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        mlp_dim=cfg.mlp_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    )
    
    total_steps = len(train_ds) * epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.learning_rate,
        decay_steps=total_steps,
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=cfg.weight_decay
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )
    model.summary()
    
    ckpt_path = cfg.ckpt_optimized
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1
        )
    ]
    
    print(f"\nBắt đầu huấn luyện {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Lưu training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = os.path.join(cfg.reports_dir, "optimized_training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"\nĐã lưu lịch sử huấn luyện vào: {hist_path}")
    print(f"Đã lưu checkpoint tốt nhất vào: {ckpt_path}")
    
    return model, history_dict


if __name__ == "__main__":
    train_optimized_model(epochs=18, batch_size=64)
