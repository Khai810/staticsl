"""
src/models/model2_hybrid_transformer_cnn/model.py
=====================================================
Kiến trúc Model 2 — Landmark-based Spatial Self-Attention Network,
lấy cảm hứng từ:

    Plizzari, C., Cannici, M., & Matteucci, M. "Spatial Temporal
    Transformer Network for Skeleton-based Action Recognition."
    ICPR Workshops 2021 (arXiv:2008.07404).

GHI CHÚ QUAN TRỌNG VỀ PHẠM VI ÁP DỤNG:
    ST-TR gốc gồm 2 module: SSA (Spatial Self-Attention — attention giữa
    các khớp TRONG CÙNG 1 frame) và TSA (Temporal Self-Attention —
    attention của CÙNG 1 khớp QUA NHIỀU FRAME theo thời gian).

    Dữ liệu hiện tại (ASL Alphabet tĩnh) chỉ có 1 frame/mẫu (T=1) — KHÔNG
    có chiều thời gian, nên TSA không áp dụng được có ý nghĩa. Kiến trúc
    dưới đây chỉ implement SSA (gọi là "S-TR" trong bài báo gốc — theo
    Table 1a của paper, riêng S-TR đã đạt 94.0% trên NTU-60 X-View, cao
    hơn cả baseline ST-GCN 92.7%, nên đây là lựa chọn hợp lý khi không có
    chiều thời gian).

    Nếu muốn dùng TRỌN VẸN cả SSA+TSA (đúng "ST-TR" đầy đủ), cần thu
    MULTI-FRAME (vài khung hình liên tiếp) cho mỗi mẫu thay vì 1 ảnh tĩnh
    — xem gợi ý mở rộng ở cuối file.

THIẾT KẾ ĐÚNG THEO ABLATION CỦA BÀI BÁO (Table 1b — "Effect of Applying
Self-Attention since Feature Extraction"): áp dụng self-attention TRỰC TIẾP
trên toạ độ thô (S-TR-all-layers) cho kết quả KÉM HƠN (93.3%) so với trích
đặc trưng bằng convolution TRƯỚC rồi mới áp SSA (94.0%). Vì vậy kiến trúc
dưới đây là TUẦN TỰ: Conv1D (đóng vai trò feature extraction, tương đương
GCN+TCN trong ST-TR gốc) -> SSA (multi-head self-attention giữa 21 khớp)
-> classification head. Đây là thay đổi so với bản trước đó (2 nhánh CNN
và Transformer chạy SONG SONG rồi fuse bằng nhân element-wise) để bám sát
đúng tinh thần ST-TR hơn.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from configs.config import HybridConfig


@tf.keras.utils.register_keras_serializable(package="model2_hybrid")
class SpatialSelfAttention(layers.Layer):
    """
    Spatial Self-Attention (SSA) — đúng công thức mục 4.2 của bài báo:

        alpha_ij = q_i . k_j^T            (tương quan giữa khớp i và khớp j)
        z_i = sum_j softmax_j(alpha_ij / sqrt(d_k)) * v_j

    KHÔNG dùng adjacency/graph cố định — để attention TỰ HỌC quan hệ giữa
    các khớp một cách động, đúng theo thiết kế gốc của Plizzari et al.
    (bài báo chủ động "discard any predefined skeleton structure").

    Multi-head: paper dùng 8 head, d_q=d_k=d_v = 0.25*C_out mỗi head.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout,
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # Self-attention giữa các khớp (query=key=value=x) + skip connection,
        # đúng như Figure "implementation" trong bài báo (BatchNorm + skip).
        attn_out = self.mha(x, x, training=training)
        return self.norm(x + attn_out)

    def get_config(self):
        # BẮT BUỘC override get_config() cho custom layer có tham số __init__
        # riêng (embed_dim, num_heads, dropout) — nếu không, model.save() vẫn
        # chạy được nhưng tf.keras.models.load_model() sau này sẽ LỖI vì
        # Keras không biết truyền lại đúng tham số khi khôi phục layer.
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="model2_hybrid")
class TransformerEncoderLayer(layers.Layer):
    """1 lớp Transformer Encoder chuẩn: Multi-Head Attention + FFN."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
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

    def get_config(self):
        # Lý do tương tự SpatialSelfAttention ở trên.
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return config


def build_landmark_hybrid_model(cfg: HybridConfig, num_classes: int) -> Model:
    """
    Xây dựng mô hình S-TR (Spatial Transformer stream) cho landmark tĩnh.

    Pipeline (TUẦN TỰ, đúng tinh thần ST-TR — xem ghi chú đầu file):
        Input (21, 3)
          -> Conv1D feature extraction (đóng vai trò "GCN+TCN" của ST-TR gốc)
          -> Chiếu về embed_dim + Positional Embedding theo chỉ số khớp
          -> N lớp SpatialSelfAttention (SSA thuần, không FFN — đúng công
             thức paper) xen kẽ TransformerEncoderLayer (SSA + FFN, để tăng
             sức biểu diễn — mở rộng thực dụng ngoài phạm vi công thức gốc)
          -> Global Average Pooling qua 21 khớp
          -> Dense -> Dropout -> Softmax
    """
    inputs = layers.Input(shape=(21, 3), name="landmark_input")

    # =====================================================================
    # FEATURE EXTRACTION: Conv1D cục bộ trước khi vào attention toàn cục
    # (theo đúng ablation của bài báo: SSA sau feature extraction > SSA
    # trực tiếp trên toạ độ thô — xem Table 1b, S-TR-all-layers).
    # =====================================================================
    x = layers.Conv1D(
        cfg.cnn_filters[0], kernel_size=3, padding="same", activation="relu", name="feature_conv1d_1"
    )(inputs)
    x = layers.Conv1D(
        cfg.cnn_filters[1], kernel_size=3, padding="same", activation="relu", name="feature_conv1d_2"
    )(x)

    # Chiếu về embed_dim để đưa vào attention.
    x = layers.Dense(cfg.vit_embed_dim, name="joint_embedding")(x)

    # Positional embedding theo CHỈ SỐ khớp (0..20) — giúp attention phân
    # biệt được "khớp nào là khớp nào" (thứ tự khớp MediaPipe cố định).
    pos_embed = layers.Embedding(
        input_dim=21, output_dim=cfg.vit_embed_dim, name="pos_embed"
    )(tf.range(start=0, limit=21, delta=1))
    x = x + pos_embed

    # =====================================================================
    # SSA STACK: các lớp Spatial Self-Attention (đúng công thức paper),
    # xen kẽ TransformerEncoderLayer (SSA + FFN) để tăng khả năng biểu diễn.
    # =====================================================================
    for i in range(cfg.vit_num_layers):
        x = SpatialSelfAttention(
            embed_dim=cfg.vit_embed_dim, num_heads=cfg.vit_num_heads,
            dropout=cfg.vit_dropout, name=f"ssa_block_{i + 1}",
        )(x)
        x = TransformerEncoderLayer(
            embed_dim=cfg.vit_embed_dim, num_heads=cfg.vit_num_heads,
            mlp_dim=cfg.vit_mlp_dim, dropout=cfg.vit_dropout,
            name=f"transformer_block_{i + 1}"
        )(x)

    pooled = layers.GlobalAveragePooling1D(name="ssa_gap")(x)

    # =====================================================================
    # CLASSIFICATION HEAD
    # =====================================================================
    x = layers.Dense(cfg.fc_hidden_dim, activation="relu", name="fc_hidden")(pooled)
    x = layers.Dropout(cfg.fc_dropout, name="fc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Landmark_STR_Model")
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