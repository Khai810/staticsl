"""
Visualization module for:
1. Skeleton plotting (MediaPipe hand connections, 5 colored finger chains)
2. Data before vs after augmentation comparison plots
3. Model training curves (Loss, Accuracy)
4. Qualitative inference samples with predictions and confidence bar charts
5. Detailed error analysis (Confusion matrix, Top confused pairs, Per-class F1)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional

# Visual styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Color scheme for 5 fingers
FINGER_COLORS = {
    'thumb': '#E53E3E',   # Red
    'index': '#3182CE',   # Blue
    'middle': '#38A169',  # Green
    'ring': '#DD6B20',    # Orange
    'pinky': '#805AD5'    # Purple
}

FINGER_CHAINS = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'pinky': [0, 17, 18, 19, 20]
}

PALM_BASE = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]


def draw_hand_skeleton_2d(ax, landmarks: np.ndarray, title: str = "", show_legend: bool = False):
    """
    Draws an anatomically colored 2D hand skeleton from landmarks of shape (21, 3).
    Note: MediaPipe Y points downward in image coordinates, so we invert Y for natural display.
    """
    xs = landmarks[:, 0]
    ys = -landmarks[:, 1] # Invert Y for natural upright view
    
    # Palm connections
    for p1, p2 in PALM_BASE:
        ax.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], color='#718096', linewidth=2.0, alpha=0.7, linestyle='--')
        
    # Draw finger bones
    for finger_name, chain in FINGER_CHAINS.items():
        color = FINGER_COLORS[finger_name]
        for i in range(len(chain) - 1):
            idx1, idx2 = chain[i], chain[i + 1]
            ax.plot([xs[idx1], xs[idx2]], [ys[idx1], ys[idx2]], color=color, linewidth=2.5, alpha=0.9, label=finger_name.capitalize() if i == 0 and show_legend else None)
            
    # Draw joint nodes
    ax.scatter(xs, ys, color='#2D3748', s=35, zorder=5, edgecolors='white', linewidths=1.0)
    # Highlight wrist
    ax.scatter([xs[0]], [ys[0]], color='#E53E3E', s=70, marker='s', zorder=6, label='Wrist (Origin)' if show_legend else None)
    
    # Fingertips (4, 8, 12, 16, 20)
    tips = [4, 8, 12, 16, 20]
    ax.scatter(xs[tips], ys[tips], color='#ECC94B', s=55, zorder=6, edgecolors='#D69E2E', linewidths=1.5, label='Fingertips' if show_legend else None)
    
    ax.set_aspect('equal', 'datalim')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=True)


def plot_augmentation_comparison(samples_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str):
    """
    Plots side-by-side comparison of hand skeletons Before vs After Light Augmentation.
    samples_dict: {letter: (original_lm, augmented_lm)}
    """
    n = len(samples_dict)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = np.array([axes])
        
    for i, (letter, (orig, aug)) in enumerate(samples_dict.items()):
        draw_hand_skeleton_2d(axes[i, 0], orig, title=f"Ký hiệu '{letter}' — Gốc (Normalized)", show_legend=(i == 0))
        draw_hand_skeleton_2d(axes[i, 1], aug, title=f"Ký hiệu '{letter}' — Sau Light Augmentation", show_legend=False)
        
    plt.suptitle("So Sánh Dữ Liệu Khung Xương Bàn Tay: Trước vs Sau Light Augmentation", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Augmentation plot saved to: {save_path}")


def plot_augmentation_distribution(orig_lm: np.ndarray, aug_list: List[np.ndarray], letter: str, save_path: str):
    """
    Overlays multiple augmented versions on top of original skeleton to visualize variability.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot augmented joints as faded points
    for aug in aug_list:
        xs = aug[:, 0]
        ys = -aug[:, 1]
        ax.scatter(xs, ys, color='#CBD5E0', s=15, alpha=0.35, zorder=2)
        
    # Draw original prominent skeleton on top
    draw_hand_skeleton_2d(ax, orig_lm, title=f"Biến thiên Không gian Điểm Khớp (Perturbation Field) — Chữ '{letter}'\n(50 mẫu Light Augmentation xếp chồng)", show_legend=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


def plot_training_curves(history_dict: Dict[str, List[float]], save_path: str, baseline_acc: float = 0.8645):
    """
    Plots training and validation loss and accuracy progression.
    """
    epochs = range(1, len(history_dict['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history_dict['loss'], 'o-', color='#3182CE', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history_dict['val_loss'], 's--', color='#E53E3E', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_title("Quá Trình Hội Tụ Hàm Mất Mát (Loss Progression)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Categorical Crossentropy (Label Smoothed)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Accuracy plot
    ax2.plot(epochs, history_dict['accuracy'], 'o-', color='#38A169', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, history_dict['val_accuracy'], 's--', color='#805AD5', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=baseline_acc, color='#D69E2E', linestyle='-.', label=f'Baseline 12-9 Test Acc ({baseline_acc*100:.1f}%)', linewidth=1.5)
    ax2.set_title("Độ Chính Xác Qua Các Epoch (Accuracy Progression)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_inference_predictions(
    test_samples: List[Tuple[np.ndarray, str, str, float, np.ndarray]],
    class_labels: List[str],
    save_path: str
):
    """
    Visualizes qualitative inference samples:
    - Left side: 2D hand skeleton
    - Right side: Top-5 prediction confidence horizontal bar chart
    Each entry: (landmarks, true_label, pred_label, confidence, prob_distribution)
    """
    n = len(test_samples)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.8 * n), gridspec_kw={'width_ratios': [1, 1.4]})
    if n == 1:
        axes = np.array([axes])
        
    for i, (lm, y_true, y_pred, conf, probs) in enumerate(test_samples):
        ax_skel = axes[i, 0]
        ax_bar = axes[i, 1]
        
        is_correct = (y_true == y_pred)
        status_color = '#38A169' if is_correct else '#E53E3E'
        title_text = f"True: '{y_true}' | Pred: '{y_pred}' ({conf*100:.1f}%)"
        
        draw_hand_skeleton_2d(ax_skel, lm, title=title_text)
        # Add border around skeleton indicating correct/wrong
        for spine in ax_skel.spines.values():
            spine.set_visible(True)
            spine.set_color(status_color)
            spine.set_linewidth(2.0)
            
        # Top-5 probabilities
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_labels = [class_labels[idx] for idx in top5_idx]
        top5_scores = probs[top5_idx] * 100.0
        
        y_pos = np.arange(len(top5_labels))
        bar_colors = [status_color if idx == top5_idx[0] else '#CBD5E0' for idx in top5_idx]
        
        ax_bar.barh(y_pos, top5_scores, color=bar_colors, edgecolor='none', height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(top5_labels, fontsize=10, fontweight='bold')
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 105)
        ax_bar.set_xlabel("Độ Tin Cậy (%)", fontsize=9)
        
        for j, score in enumerate(top5_scores):
            ax_bar.text(score + 1.5, j, f"{score:.1f}%", va='center', fontsize=9, fontweight='bold' if j == 0 else 'normal')
            
        ax_bar.grid(axis='x', linestyle=':', alpha=0.5)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        
    plt.suptitle("Trực Quan Hóa Kết Quả Dự Đoán & Điểm Tin Cậy (Confidence) Trên Tập Test", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Inference predictions plot saved to: {save_path}")


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_labels: List[str], save_path: str, title: str = "Confusion Matrix"):
    """Plots clean annotated confusion matrix heatmap."""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar_kws={'label': 'Số lượng mẫu'})
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=12, labelpad=10)
    plt.ylabel('Nhãn Thực Tế (Ground Truth)', fontsize=12, labelpad=10)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_per_class_f1(report_dict: dict, class_labels: List[str], save_path: str):
    """Plots per-class F1-scores as a horizontal bar chart."""
    f1s = [report_dict[cls]['f1-score'] for cls in class_labels if cls in report_dict]
    cls_names = [cls for cls in class_labels if cls in report_dict]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(cls_names))
    colors = ['#38A169' if score >= 0.90 else '#DD6B20' if score >= 0.75 else '#E53E3E' for score in f1s]
    
    plt.barh(y_pos, f1s, color=colors, height=0.65)
    plt.yticks(y_pos, cls_names, fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.05)
    plt.axvline(x=0.90, color='#38A169', linestyle='--', alpha=0.7, label='Ngưỡng Xuất Sắc (90%)')
    plt.axvline(x=0.75, color='#DD6B20', linestyle='--', alpha=0.7, label='Ngưỡng Chấp Nhận Được (75%)')
    plt.xlabel('F1-Score', fontsize=11)
    plt.title('Điểm F1-Score Chi Tiết Cho Từng Ký Tự ASL (Tập Test)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    
    for i, v in enumerate(f1s):
        plt.text(v + 0.015, i, f"{v:.2f}", va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Per-class F1 plot saved to: {save_path}")


def plot_model_comparison_bar(models_metrics: Dict[str, Dict[str, float]], save_path: str):
    """
    Plots comprehensive comparative metrics (Accuracy, Macro F1, Weighted F1, Top-3 Acc) across models.
    """
    model_names = list(models_metrics.keys())
    metric_keys = ['accuracy', 'macro_f1', 'weighted_f1']
    metric_labels = ['Test Accuracy', 'Macro F1-Score', 'Weighted F1-Score']
    
    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ['#4A5568', '#E53E3E', '#3182CE']
    
    plt.figure(figsize=(10, 5.5))
    for i, m_name in enumerate(model_names):
        vals = [models_metrics[m_name][m] for m in metric_keys]
        bars = plt.bar(x + (i - 1) * width, vals, width, label=m_name, color=colors[i % len(colors)], edgecolor='white')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.015, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.ylabel('Tỉ Lệ Đạt Được (%)', fontsize=11)
    plt.title('So Sánh Tổng Thể Hiệu Năng 3 Mô Hình Trên Tập Test Độc Lập', fontsize=13, fontweight='bold')
    plt.xticks(x, metric_labels, fontsize=11)
    plt.ylim(0, 1.12)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Model comparison bar plot saved to: {save_path}")
