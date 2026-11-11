"""
Model architectures module:
1. Baseline 12-9: Dual-branch CNN + ViT with Element-wise Multiply Fusion
2. S-TR 14-9: Sequential Conv1D + Spatial Self Attention Stack
3. Optimized ST-Hybrid: Topology-Aware Spatial Transformer with Gated Finger Kinematics
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

# =====================================================================
# 1. BASELINE 12-9 LAYERS & MODEL
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="slr_models")
class BaselineTransformerEncoderLayer(layers.Layer):
    """Standard Transformer Encoder Layer for 12-9."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_dim: int = 256, dropout: float = 0.1, **kwargs):
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
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return cfg


def build_model_12_9(num_classes: int = 29, embed_dim: int = 128, num_heads: int = 4, mlp_dim: int = 256, num_layers: int = 2, dropout: float = 0.1) -> Model:
    """Reconstructs exact Model 12-9 architecture."""
    inputs = layers.Input(shape=(21, 3), name="landmark_input")
    
    # Branch 1: 1D-CNN
    x_cnn = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", name="cnn1d_1")(inputs)
    x_cnn = layers.MaxPooling1D(pool_size=2, name="pool1d_1")(x_cnn)
    x_cnn = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="cnn1d_2")(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D(name="cnn_gap")(x_cnn)
    cnn_vector = layers.Dense(embed_dim, activation="relu", name="cnn_dense")(x_cnn)
    
    # Branch 2: Spatial Transformer
    x_vit = layers.Dense(embed_dim, name="joint_embedding")(inputs)
    pos_embed = layers.Embedding(input_dim=21, output_dim=embed_dim, name="pos_embed")(tf.range(start=0, limit=21, delta=1))
    x_vit = x_vit + pos_embed
    for i in range(num_layers):
        x_vit = BaselineTransformerEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout,
            name=f"transformer_block_{i + 1}"
        )(x_vit)
    vit_vector = layers.GlobalAveragePooling1D(name="vit_gap")(x_vit)
    
    # Element-wise multiply fusion
    fused = layers.Multiply(name="feature_fusion_elementwise")([cnn_vector, vit_vector])
    x = layers.Dense(128, activation="relu", name="fc_hidden")(fused)
    x = layers.Dropout(0.3, name="fc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="Landmark_Hybrid_Model_12_9")


# =====================================================================
# 2. SKELETON 14-9 LAYERS & MODEL
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="slr_models")
class SpatialSelfAttention(layers.Layer):
    """Pure Spatial Self-Attention without FFN, from Plizzari et al. (ST-TR)."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        return self.norm(x + attn_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "dropout": self.dropout})
        return cfg


def build_model_14_9(num_classes: int = 29, embed_dim: int = 128, num_heads: int = 4, mlp_dim: int = 256, num_layers: int = 2, dropout: float = 0.1) -> Model:
    """Reconstructs exact Model 14-9 architecture."""
    inputs = layers.Input(shape=(21, 3), name="landmark_input")
    
    # Sequential Conv1D feature extraction
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", name="feature_conv1d_1")(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="feature_conv1d_2")(x)
    x = layers.Dense(embed_dim, name="joint_embedding")(x)
    
    pos_embed = layers.Embedding(input_dim=21, output_dim=embed_dim, name="pos_embed")(tf.range(start=0, limit=21, delta=1))
    x = x + pos_embed
    
    for i in range(num_layers):
        x = SpatialSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, name=f"ssa_block_{i + 1}")(x)
        x = BaselineTransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, name=f"transformer_block_{i + 1}")(x)
        
    pooled = layers.GlobalAveragePooling1D(name="ssa_gap")(x)
    x = layers.Dense(128, activation="relu", name="fc_hidden")(pooled)
    x = layers.Dropout(0.3, name="fc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="Landmark_STR_Model_14_9")


# =====================================================================
# 3. OPTIMIZED TOPOLOGY-AWARE HYBRID MODEL (ST-Hybrid)
# =====================================================================

@tf.keras.utils.register_keras_serializable(package="slr_models")
class GatedFusion(layers.Layer):
    """Adaptive gated fusion balancing local kinematics and global spatial attention."""
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.gate_dense = layers.Dense(embed_dim, activation="sigmoid")
        self.proj = layers.Dense(embed_dim, activation="gelu")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x_local, x_global = inputs
        gate = self.gate_dense(tf.concat([x_local, x_global], axis=-1))
        fused = gate * x_local + (1.0 - gate) * x_global
        out = self.proj(fused)
        return self.norm(out + fused)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim})
        return cfg


def build_optimized_st_hybrid_model(num_classes: int = 29, embed_dim: int = 128, num_heads: int = 4, mlp_dim: int = 256, num_layers: int = 2, dropout: float = 0.15) -> Model:
    """
    Optimized Topology-Aware Spatial-Temporal Skeleton Hybrid Model:
    1. Input: (21, 3) normalized joints + optional orientation feature
    2. Finger Topology Embedding: Joint Index (0..20) + Finger ID (0..5)
    3. Stream A: Local Finger Kinematics (Conv1D + Residual with kernel 3)
    4. Stream B: Global Spatial Self-Attention (Pre-LN Transformer Blocks)
    5. Gated Adaptive Fusion (avoids zeroing-out from element-wise multiplication)
    6. Orientation Awareness: Injects palm direction vector to retain hand tilt
    """
    inputs = layers.Input(shape=(21, 3), name="landmark_input")
    
    # 1. Embed joint coordinates
    x_emb = layers.Dense(embed_dim, activation="linear", name="joint_proj")(inputs)
    
    # 2. Joint index embedding (0..20)
    joint_indices = tf.range(0, 21, delta=1)
    pos_emb = layers.Embedding(input_dim=21, output_dim=embed_dim, name="joint_pos_emb")(joint_indices)
    
    # 3. Finger membership embedding:
    # 0: Wrist, 1: Thumb, 2: Index, 3: Middle, 4: Ring, 5: Pinky
    finger_ids = tf.constant([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5], dtype=tf.int32)
    finger_emb = layers.Embedding(input_dim=6, output_dim=embed_dim, name="finger_topo_emb")(finger_ids)
    
    # Combined token representation
    x = x_emb + pos_emb + finger_emb
    
    # Stream A: Local Kinematic Conv Block
    conv1 = layers.Conv1D(embed_dim, kernel_size=3, padding="same", activation="gelu", name="local_conv1")(x)
    conv1 = layers.LayerNormalization(epsilon=1e-6)(conv1 + x)
    conv2 = layers.Conv1D(embed_dim, kernel_size=3, padding="same", activation="gelu", name="local_conv2")(conv1)
    local_stream = layers.LayerNormalization(epsilon=1e-6)(conv2 + conv1)
    local_vec = layers.GlobalAveragePooling1D(name="local_gap")(local_stream)
    
    # Stream B: Global Spatial Transformer Block (Pre-LN)
    global_stream = x
    for i in range(num_layers):
        global_stream = BaselineTransformerEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout,
            name=f"spatial_attn_layer_{i+1}"
        )(global_stream)
    global_vec = layers.GlobalAveragePooling1D(name="global_gap")(global_stream)
    
    # Stream C: Global Hand Orientation Feature
    # Orientation vector is difference between middle MCP (index 9) and wrist (index 0)
    wrist_pt = inputs[:, 0, :]
    mid_mcp = inputs[:, 9, :]
    orient_vec = mid_mcp - wrist_pt
    orient_feat = layers.Dense(32, activation="gelu", name="orient_proj")(orient_vec)
    
    # Gated Fusion of Local and Global Streams
    fused_vec = GatedFusion(embed_dim=embed_dim, name="gated_fusion")([local_vec, global_vec])
    
    # Combine with orientation feature
    full_rep = layers.Concatenate(name="rep_concat")([fused_vec, orient_feat])
    
    # Classification Head
    h = layers.Dense(128, activation="gelu", name="fc_rep")(full_rep)
    h = layers.LayerNormalization(epsilon=1e-6)(h)
    h = layers.Dropout(dropout, name="head_dropout")(h)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(h)
    
    model = Model(inputs=inputs, outputs=outputs, name="Optimized_ST_Hybrid_Model")
    return model


def load_weights_12_9(model: Model, zip_path: str):
    """Loads weights from checkpoint model2_hybrid_transformer_cnn_best_12-9.keras."""
    import zipfile, h5py, io
    _ = model(tf.zeros((1, 21, 3)))
    with zipfile.ZipFile(zip_path) as z:
        with h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r') as h:
            l = h['layers']
            model.get_layer('cnn1d_1').set_weights([l['conv1d']['vars']['0'][:], l['conv1d']['vars']['1'][:]])
            model.get_layer('cnn1d_2').set_weights([l['conv1d_1']['vars']['0'][:], l['conv1d_1']['vars']['1'][:]])
            model.get_layer('joint_embedding').set_weights([l['dense']['vars']['0'][:], l['dense']['vars']['1'][:]])
            model.get_layer('cnn_dense').set_weights([l['dense_1']['vars']['0'][:], l['dense_1']['vars']['1'][:]])
            model.get_layer('fc_hidden').set_weights([l['dense_2']['vars']['0'][:], l['dense_2']['vars']['1'][:]])
            model.get_layer('classification_head').set_weights([l['dense_3']['vars']['0'][:], l['dense_3']['vars']['1'][:]])
            
            for blk_name, h5_name in [('transformer_block_1', 'transformer_encoder_layer'),
                                      ('transformer_block_2', 'transformer_encoder_layer_1')]:
                layer = model.get_layer(blk_name)
                hl = l[h5_name]
                layer.attn.set_weights([
                    hl['attn']['query_dense']['vars']['0'][:], hl['attn']['query_dense']['vars']['1'][:],
                    hl['attn']['key_dense']['vars']['0'][:], hl['attn']['key_dense']['vars']['1'][:],
                    hl['attn']['value_dense']['vars']['0'][:], hl['attn']['value_dense']['vars']['1'][:],
                    hl['attn']['output_dense']['vars']['0'][:], hl['attn']['output_dense']['vars']['1'][:]
                ])
                layer.norm1.set_weights([hl['norm1']['vars']['0'][:], hl['norm1']['vars']['1'][:]])
                layer.mlp.layers[0].set_weights([hl['mlp']['layers']['dense']['vars']['0'][:], hl['mlp']['layers']['dense']['vars']['1'][:]])
                layer.mlp.layers[2].set_weights([hl['mlp']['layers']['dense_1']['vars']['0'][:], hl['mlp']['layers']['dense_1']['vars']['1'][:]])
                layer.norm2.set_weights([hl['norm2']['vars']['0'][:], hl['norm2']['vars']['1'][:]])


def load_weights_14_9(model: Model, zip_path: str):
    """Loads weights from checkpoint model2_hybrid_transformer_cnn_best14-9.keras."""
    import zipfile, h5py, io
    _ = model(tf.zeros((1, 21, 3)))
    with zipfile.ZipFile(zip_path) as z:
        with h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r') as h:
            l = h['layers']
            model.get_layer('feature_conv1d_1').set_weights([l['conv1d']['vars']['0'][:], l['conv1d']['vars']['1'][:]])
            model.get_layer('feature_conv1d_2').set_weights([l['conv1d_1']['vars']['0'][:], l['conv1d_1']['vars']['1'][:]])
            model.get_layer('joint_embedding').set_weights([l['dense']['vars']['0'][:], l['dense']['vars']['1'][:]])
            model.get_layer('fc_hidden').set_weights([l['dense_1']['vars']['0'][:], l['dense_1']['vars']['1'][:]])
            model.get_layer('classification_head').set_weights([l['dense_2']['vars']['0'][:], l['dense_2']['vars']['1'][:]])
            
            for blk_name, h5_name in [('ssa_block_1', 'spatial_self_attention'), ('ssa_block_2', 'spatial_self_attention_1')]:
                layer = model.get_layer(blk_name)
                hl = l[h5_name]
                layer.mha.set_weights([
                    hl['mha']['query_dense']['vars']['0'][:], hl['mha']['query_dense']['vars']['1'][:],
                    hl['mha']['key_dense']['vars']['0'][:], hl['mha']['key_dense']['vars']['1'][:],
                    hl['mha']['value_dense']['vars']['0'][:], hl['mha']['value_dense']['vars']['1'][:],
                    hl['mha']['output_dense']['vars']['0'][:], hl['mha']['output_dense']['vars']['1'][:]
                ])
                layer.norm.set_weights([hl['norm']['vars']['0'][:], hl['norm']['vars']['1'][:]])
                
            for blk_name, h5_name in [('transformer_block_1', 'transformer_encoder_layer'), ('transformer_block_2', 'transformer_encoder_layer_1')]:
                layer = model.get_layer(blk_name)
                hl = l[h5_name]
                layer.attn.set_weights([
                    hl['attn']['query_dense']['vars']['0'][:], hl['attn']['query_dense']['vars']['1'][:],
                    hl['attn']['key_dense']['vars']['0'][:], hl['attn']['key_dense']['vars']['1'][:],
                    hl['attn']['value_dense']['vars']['0'][:], hl['attn']['value_dense']['vars']['1'][:],
                    hl['attn']['output_dense']['vars']['0'][:], hl['attn']['output_dense']['vars']['1'][:]
                ])
                layer.norm1.set_weights([hl['norm1']['vars']['0'][:], hl['norm1']['vars']['1'][:]])
                layer.mlp.layers[0].set_weights([hl['mlp']['layers']['dense']['vars']['0'][:], hl['mlp']['layers']['dense']['vars']['1'][:]])
                layer.mlp.layers[2].set_weights([hl['mlp']['layers']['dense_1']['vars']['0'][:], hl['mlp']['layers']['dense_1']['vars']['1'][:]])
                layer.norm2.set_weights([hl['norm2']['vars']['0'][:], hl['norm2']['vars']['1'][:]])

