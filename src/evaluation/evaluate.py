"""
src/evaluation/evaluate.py
==========================
Đánh giá toàn diện 3 mô hình nhận diện ngôn ngữ ký hiệu ASL:
1. Baseline Model 12-9 (Dual-branch CNN + ViT)
2. S-TR Model 14-9 (Sequential Conv1D + Spatial Attention Stack - Lỗi xoay)
3. Optimized ST-Hybrid Model (Topology Embedding + Gated Fusion + Orientation Stream)

Xuất ra:
- Ma trận nhầm lẫn (Confusion Matrix) cho cả 3 mô hình
- Biểu đồ F1-score từng ký tự (Per-class F1)
- Biểu đồ so sánh Accuracy / Macro F1
- Minh họa trực quan nhận diện kèm độ tin cậy (Confidence)
- Phân tích Top cặp ký tự dễ nhầm lẫn nhất (Top Confused Pairs)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Hỗ trợ chạy trực tiếp từ bất kỳ thư mục nào
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import PreprocessConfig
from src.models.st_hybrid import (
    build_model_12_9, build_model_14_9, build_optimized_st_hybrid_model,
    load_weights_12_9, load_weights_14_9
)
from src.datasets.dataset import FlexibleLandmarkDataset
from src.preprocessing.normalization import normalize_12_9
from src.preprocessing.augmentation import augment_landmarks_light
from src.utils.visualization import (
    plot_augmentation_comparison,
    plot_augmentation_distribution,
    plot_training_curves,
    plot_inference_predictions,
    plot_confusion_matrix_heatmap,
    plot_per_class_f1,
    plot_model_comparison_bar
)


def run_full_evaluation():
    cfg = PreprocessConfig()
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)
    
    print("=" * 65)
    print("1. XUẤT BIỂU ĐỒ MINH HỌA DATA AUGMENTATION (TRƯỚC & SAU)...")
    print("=" * 65)
    
    test_df = pd.read_csv(cfg.test_manifest)
    feature_cols = [f"r{i}" for i in range(63)]
    
    selected_letters = ['A', 'B', 'C', 'L', 'V', 'Y']
    samples_dict = {}
    for letter in selected_letters:
        sub = test_df[test_df['label'] == letter]
        if len(sub) > 0:
            raw_lm = sub[feature_cols].iloc[0].values.astype(np.float32).reshape(21, 3)
            norm_lm = normalize_12_9(raw_lm)
            aug_lm = augment_landmarks_light(norm_lm, rotation_range_deg=12.0, jitter_std=0.008)
            samples_dict[letter] = (norm_lm, aug_lm)
            
    aug_comp_path = os.path.join(cfg.plots_dir, "01_augmentation_comparison.png")
    plot_augmentation_comparison(samples_dict, aug_comp_path)
    
    if 'L' in samples_dict:
        base_L = samples_dict['L'][0]
        rng = np.random.default_rng(42)
        aug_list = [augment_landmarks_light(base_L, rotation_range_deg=12.0, jitter_std=0.008, rng=rng) for _ in range(50)]
        dist_path = os.path.join(cfg.plots_dir, "02_augmentation_distribution.png")
        plot_augmentation_distribution(base_L, aug_list, 'L', dist_path)
        
    print("\n" + "=" * 65)
    print("2. ĐÁNH GIÁ BASELINE MODEL 12-9 TRÊN TEST SET...")
    print("=" * 65)
    
    m12 = build_model_12_9(num_classes=cfg.num_classes)
    load_weights_12_9(m12, cfg.ckpt_12_9)
    m12.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    test_ds_12 = FlexibleLandmarkDataset(cfg.test_manifest, cfg.class_labels, norm_mode='12_9', augment=False, shuffle=False)
    y_true_12 = []
    for i in range(len(test_ds_12)):
        _, yb = test_ds_12[i]
        y_true_12.extend(np.argmax(yb, axis=1))
    y_true_12 = np.array(y_true_12)
    
    y_pred_probs_12 = m12.predict(test_ds_12, verbose=1)
    y_pred_12 = np.argmax(y_pred_probs_12, axis=1)
    
    loss_12, acc_12 = m12.evaluate(test_ds_12, verbose=0)
    macro_f1_12 = f1_score(y_true_12, y_pred_12, average='macro', zero_division=0)
    weighted_f1_12 = f1_score(y_true_12, y_pred_12, average='weighted', zero_division=0)
    print(f"-> Model 12-9: Loss = {loss_12:.4f}, Accuracy = {acc_12:.4f}, Macro F1 = {macro_f1_12:.4f}")
    
    cm_12 = confusion_matrix(y_true_12, y_pred_12, labels=range(cfg.num_classes))
    plot_confusion_matrix_heatmap(
        cm_12, cfg.class_labels,
        os.path.join(cfg.plots_dir, "03_confusion_matrix_model12_9.png"),
        title="Ma Trận Nhầm Lẫn (Confusion Matrix) — Baseline Model 12-9 (Test Acc: 79.5%)"
    )
    
    print("\n" + "=" * 65)
    print("3. ĐÁNH GIÁ S-TR MODEL 14-9 (BỊ LỖI CANONICALIZE ROTATION)...")
    print("=" * 65)
    
    m14 = build_model_14_9(num_classes=cfg.num_classes)
    load_weights_14_9(m14, cfg.ckpt_14_9)
    m14.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    test_ds_14 = FlexibleLandmarkDataset(cfg.test_manifest, cfg.class_labels, norm_mode='14_9', augment=False, shuffle=False)
    y_pred_probs_14 = m14.predict(test_ds_14, verbose=1)
    y_pred_14 = np.argmax(y_pred_probs_14, axis=1)
    
    loss_14, acc_14 = m14.evaluate(test_ds_14, verbose=0)
    macro_f1_14 = f1_score(y_true_12, y_pred_14, average='macro', zero_division=0)
    weighted_f1_14 = f1_score(y_true_12, y_pred_14, average='weighted', zero_division=0)
    print(f"-> Model 14-9: Loss = {loss_14:.4f}, Accuracy = {acc_14:.4f}, Macro F1 = {macro_f1_14:.4f}")
    
    cm_14 = confusion_matrix(y_true_12, y_pred_14, labels=range(cfg.num_classes))
    plot_confusion_matrix_heatmap(
        cm_14, cfg.class_labels,
        os.path.join(cfg.plots_dir, "04_confusion_matrix_model14_9.png"),
        title="Ma Trận Nhầm Lẫn — S-TR Model 14-9 (Test Acc: 20.6% - Rotation Error)"
    )
    
    print("\n" + "=" * 65)
    print("4. ĐÁNH GIÁ OPTIMIZED ST-HYBRID MODEL...")
    print("=" * 65)
    
    import keras
    import src.models.st_hybrid
    if os.path.exists(cfg.ckpt_optimized):
        print(f"Load checkpoint tối ưu từ: {cfg.ckpt_optimized}")
        m_opt = keras.models.load_model(cfg.ckpt_optimized)
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint {cfg.ckpt_optimized}!")
        m_opt = build_optimized_st_hybrid_model(num_classes=cfg.num_classes)
        m_opt.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    test_ds_opt = FlexibleLandmarkDataset(cfg.test_manifest, cfg.class_labels, norm_mode='12_9', augment=False, shuffle=False)
    y_pred_probs_opt = m_opt.predict(test_ds_opt, verbose=1)
    y_pred_opt = np.argmax(y_pred_probs_opt, axis=1)
    
    loss_opt, acc_opt = m_opt.evaluate(test_ds_opt, verbose=0)
    macro_f1_opt = f1_score(y_true_12, y_pred_opt, average='macro', zero_division=0)
    weighted_f1_opt = f1_score(y_true_12, y_pred_opt, average='weighted', zero_division=0)
    print(f"-> Optimized Model: Loss = {loss_opt:.4f}, Accuracy = {acc_opt:.4f}, Macro F1 = {macro_f1_opt:.4f}")
    
    cm_opt = confusion_matrix(y_true_12, y_pred_opt, labels=range(cfg.num_classes))
    plot_confusion_matrix_heatmap(
        cm_opt, cfg.class_labels,
        os.path.join(cfg.plots_dir, "05_confusion_matrix_optimized.png"),
        title=f"Ma Trận Nhầm Lẫn — Optimized ST-Hybrid (Test Acc: {acc_opt*100:.1f}%)"
    )
    
    rep_opt = classification_report(y_true_12, y_pred_opt, labels=range(cfg.num_classes), target_names=cfg.class_labels, output_dict=True, zero_division=0)
    plot_per_class_f1(rep_opt, cfg.class_labels, os.path.join(cfg.plots_dir, "06_per_class_f1_optimized.png"))
    
    # 5. Training curves plot
    hist_path = os.path.join(cfg.reports_dir, "optimized_training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            hist_dict = json.load(f)
        plot_training_curves(hist_dict, os.path.join(cfg.plots_dir, "07_training_curves_optimized.png"), baseline_acc=acc_12)
        
    # 6. Overall Comparison bar chart
    comparison_dict = {
        "Baseline 12-9": {"accuracy": float(acc_12), "macro_f1": float(macro_f1_12), "weighted_f1": float(weighted_f1_12)},
        "S-TR 14-9": {"accuracy": float(acc_14), "macro_f1": float(macro_f1_14), "weighted_f1": float(weighted_f1_14)},
        "Optimized ST-Hybrid": {"accuracy": float(acc_opt), "macro_f1": float(macro_f1_opt), "weighted_f1": float(weighted_f1_opt)}
    }
    plot_model_comparison_bar(comparison_dict, os.path.join(cfg.plots_dir, "08_model_comparison_metrics.png"))
    
    # 7. Qualitative Inference predictions
    print("\n" + "=" * 65)
    print("5. XUẤT ẢNH TRỰC QUAN HÓA DỰ ĐOÁN & CONFIDENCE...")
    print("=" * 65)
    
    demo_indices = [15, 120, 350, 780, 1420, 2100]
    test_samples_vis = []
    
    for idx in demo_indices:
        b_idx = idx // 64
        item_idx = idx % 64
        X_b, y_b = test_ds_opt[b_idx]
        lm_sample = X_b[item_idx]
        true_cls_idx = np.argmax(y_b[item_idx])
        true_label = cfg.class_labels[true_cls_idx]
        
        probs = y_pred_probs_opt[idx]
        pred_cls_idx = np.argmax(probs)
        pred_label = cfg.class_labels[pred_cls_idx]
        conf = float(probs[pred_cls_idx])
        
        test_samples_vis.append((lm_sample, true_label, pred_label, conf, probs))
        
    plot_inference_predictions(test_samples_vis, cfg.class_labels, os.path.join(cfg.plots_dir, "09_qualitative_inference_predictions.png"))
    
    # 8. Compute Top Confused Pairs
    confused_pairs_12 = []
    for r in range(len(cfg.class_labels)):
        for c in range(len(cfg.class_labels)):
            if r != c and cm_12[r, c] > 0:
                confused_pairs_12.append({
                    "true_class": cfg.class_labels[r],
                    "pred_class": cfg.class_labels[c],
                    "count": int(cm_12[r, c]),
                    "true_total": int(np.sum(cm_12[r, :])),
                    "rate_pct": float(cm_12[r, c] / np.sum(cm_12[r, :]) * 100)
                })
    confused_pairs_12.sort(key=lambda x: x["count"], reverse=True)
    
    confused_pairs_opt = []
    for r in range(len(cfg.class_labels)):
        for c in range(len(cfg.class_labels)):
            if r != c and cm_opt[r, c] > 0:
                confused_pairs_opt.append({
                    "true_class": cfg.class_labels[r],
                    "pred_class": cfg.class_labels[c],
                    "count": int(cm_opt[r, c]),
                    "true_total": int(np.sum(cm_opt[r, :])),
                    "rate_pct": float(cm_opt[r, c] / np.sum(cm_opt[r, :]) * 100)
                })
    confused_pairs_opt.sort(key=lambda x: x["count"], reverse=True)
    
    full_results = {
        "models_summary": comparison_dict,
        "classification_reports": {
            "baseline_12_9": classification_report(y_true_12, y_pred_12, labels=range(cfg.num_classes), target_names=cfg.class_labels, output_dict=True, zero_division=0),
            "str_14_9": classification_report(y_true_12, y_pred_14, labels=range(cfg.num_classes), target_names=cfg.class_labels, output_dict=True, zero_division=0),
            "optimized_st_hybrid": rep_opt
        },
        "top_confused_pairs_12_9": confused_pairs_12[:10],
        "top_confused_pairs_optimized": confused_pairs_opt[:10]
    }
    
    results_path = os.path.join(cfg.reports_dir, "evaluation_metrics_full.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nHoàn tất toàn bộ đánh giá! Đã lưu metrics vào {results_path}")


if __name__ == "__main__":
    run_full_evaluation()
