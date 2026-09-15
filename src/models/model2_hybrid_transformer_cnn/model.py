"""
src/models/model2_hybrid_transformer_cnn/model.py
=====================================================
Kiến trúc Model 2 — Landmark-based Hybrid Transformer-CNN.

Lấy cảm hứng từ bài báo của Aly & Fathi (2025) nhưng được tối ưu hoá 
để nhận đầu vào là toạ độ điểm khớp (Landmarks) thay vì ảnh thô:
    - Input: Tensor shape (21, 3) tương ứng 21 khớp x 3 toạ độ (x,y,z).
    - Nhánh 1 (1D-CNN): Trích xuất đặc trưng cục bộ giữa các khớp liền kề.
    - Nhánh 2 (Spatial Transformer): Trích xuất tương quan hình học toàn cục.
    - Fusion: Nhân element-wise giữa 2 nhánh.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.config import HybridConfig


class TransformerEncoderLayer(layers.Layer):
    """1 lớp Transformer Encoder chuẩn: Multi-Head Attention + FFN."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.attn_dropout = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.mlp_dropout = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        attn_out = self.attn_dropout(attn_out, training=training)
        x = self.norm1(x + attn_out)

        mlp_out = self.mlp(x, training=training)
        mlp_out = self.mlp_dropout(mlp_out, training=training)
        x = self.norm2(x + mlp_out)
        return x


def build_landmark_hybrid_model(cfg: HybridConfig, num_classes: int) -> Model:
    """
    Xây dựng mô hình Landmark-based Hybrid Transformer-CNN.
    """
    # Đầu vào là toạ độ 21 khớp bàn tay (x, y, z)
    inputs = layers.Input(shape=(21, 3), name="landmark_input")

    # =====================================================================
    # NHÁNH 1: 1D-CNN (Trích xuất đặc trưng hình thái cục bộ)
    # =====================================================================
    x_cnn = layers.Conv1D(
        cfg.cnn_filters[0], kernel_size=3, padding="same", activation="relu", name="cnn1d_1"
    )(inputs)
    x_cnn = layers.MaxPooling1D(pool_size=2, name="pool1d_1")(x_cnn)

    x_cnn = layers.Conv1D(
        cfg.cnn_filters[1], kernel_size=3, padding="same", activation="relu", name="cnn1d_2"
    )(x_cnn)

    x_cnn = layers.GlobalAveragePooling1D(name="cnn_gap")(x_cnn)
    cnn_vector = layers.Dense(cfg.vit_embed_dim, activation="relu", name="cnn_dense")(x_cnn)

    # =====================================================================
    # NHÁNH 2: SPATIAL TRANSFORMER (Học tương quan toàn cục giữa 21 khớp)
    # =====================================================================
    # Chiếu mỗi khớp (3 toạ độ) lên không gian embedding
    x_vit = layers.Dense(cfg.vit_embed_dim, name="joint_embedding")(inputs)

    # Positional encoding cho 21 khớp (để model biết vị trí tương đối của từng khớp)
    pos_embed = layers.Embedding(
        input_dim=21, output_dim=cfg.vit_embed_dim, name="pos_embed"
    )(tf.range(start=0, limit=21, delta=1))
    x_vit = x_vit + pos_embed

    # Các lớp Transformer Encoder
    for i in range(cfg.vit_num_layers):
        x_vit = TransformerEncoderLayer(
            embed_dim=cfg.vit_embed_dim, num_heads=cfg.vit_num_heads,
            mlp_dim=cfg.vit_mlp_dim, dropout=cfg.vit_dropout,
            name=f"transformer_block_{i + 1}"
        )(x_vit)

    vit_vector = layers.GlobalAveragePooling1D(name="vit_gap")(x_vit)

    # =====================================================================
    # FUSION & CLASSIFICATION
    # =====================================================================
    # Fusion bằng phép nhân Element-wise theo bài báo gốc
    fused = layers.Multiply(name="feature_fusion_elementwise")([cnn_vector, vit_vector])

    x = layers.Dense(cfg.fc_hidden_dim, activation="relu", name="fc_hidden")(fused)
    x = layers.Dropout(cfg.fc_dropout, name="fc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Landmark_Hybrid_Model")
    return model


def compile_model(model: Model, cfg: HybridConfig, steps_per_epoch: int) -> Model:
    """Compile model với cấu hình tối ưu AdamW và Cosine Decay."""
    total_steps = steps_per_epoch * cfg.epochs

    if cfg.lr_schedule == "cosine_decay":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate, decay_steps=total_steps,
        )
    else:
        lr_schedule = cfg.learning_rate

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model