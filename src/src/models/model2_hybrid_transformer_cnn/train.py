"""
src/models/model2_hybrid_transformer_cnn/train.py
=====================================================
Script huấn luyện Model 2 (Landmark-based Hybrid Transformer-CNN).
"""

import argparse
import os
import tensorflow as tf
from configs.config import PreprocessConfig
from src.models.model2_hybrid_transformer_cnn.dataset import LandmarkHybridDataset
from src.models.model2_hybrid_transformer_cnn.model import build_landmark_hybrid_model, compile_model

def _parse_args() -> argparse.Namespace:
    cfg = PreprocessConfig()
    parser = argparse.ArgumentParser(description="Huấn luyện Model 2 - Landmark Hybrid Transformer-CNN.")
    parser.add_argument("--train_manifest", type=str, default=cfg.hybrid_train_manifest)
    parser.add_argument("--val_manifest", type=str, default=cfg.hybrid_val_manifest)
    parser.add_argument("--checkpoint_path", type=str, default=cfg.hybrid_checkpoint_path)
    parser.add_argument("--epochs", type=int, default=cfg.hybrid.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.hybrid.batch_size)
    parser.add_argument("--learning_rate", type=float, default=cfg.hybrid.learning_rate)
    parser.add_argument("--early_stopping_patience", type=int, default=cfg.hybrid.early_stopping_patience)
    return parser.parse_args()

def main():
    args = _parse_args()
    cfg = PreprocessConfig()
    cfg.hybrid.epochs = args.epochs
    cfg.hybrid.batch_size = args.batch_size
    cfg.hybrid.learning_rate = args.learning_rate
    cfg.hybrid.early_stopping_patience = args.early_stopping_patience

    class_labels = cfg.hybrid.hybrid_class_labels
    num_classes = len(class_labels)

    train_ds = LandmarkHybridDataset(args.train_manifest, cfg, class_labels, training=True)
    val_ds = LandmarkHybridDataset(args.val_manifest, cfg, class_labels, training=False)

    model = build_landmark_hybrid_model(cfg.hybrid, num_classes)
    model = compile_model(model, cfg.hybrid, steps_per_epoch=len(train_ds))
    model.summary()

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.hybrid.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.hybrid.epochs,
        callbacks=callbacks,
    )

    train_ds.close()
    val_ds.close()
    print(f"\n[train] Đã lưu model tốt nhất tại: {args.checkpoint_path}")

if __name__ == "__main__":
    main()