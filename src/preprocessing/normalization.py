"""
Landmark Normalization module:
Contains baseline 12-9 normalization, problematic 14-9 rotation canonicalization,
and optimized robust feature extraction (landmarks, bone vectors, angles, orientation).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

def normalize_12_9(landmarks: np.ndarray, origin_idx: int = 0, ref_pair: Tuple[int, int] = (0, 9)) -> np.ndarray:
    """
    Normalization used in Skeleton12_9:
    1. Translate wrist (origin_idx=0) to (0,0,0)
    2. Scale by Euclidean distance between origin (0) and middle MCP (9)
    Keeps hand orientation in space (CRITICAL for ASL letters like D vs G, K vs P).
    """
    origin = landmarks[origin_idx]
    translated = landmarks - origin
    
    idx1, idx2 = ref_pair
    ref_distance = np.linalg.norm(translated[idx1] - translated[idx2])
    scaled = translated / ref_distance if ref_distance > 1e-6 else translated
    return scaled.astype(np.float32)


def canonicalize_rotation_14_9(landmarks_translated: np.ndarray, ref_pair: Tuple[int, int] = (0, 9)) -> np.ndarray:
    """
    The rotation canonicalization introduced in Skeleton14_9.
    Rotates the hand in the XY plane so that vector (0 -> 9) points to +Y.
    NOTE: This removes global orientation, destroying distinctions like D vs G, P vs K, Q vs G!
    """
    idx1, idx2 = ref_pair
    vx = landmarks_translated[idx2, 0] - landmarks_translated[idx1, 0]
    vy = landmarks_translated[idx2, 1] - landmarks_translated[idx1, 1]
    theta = np.arctan2(vx, vy)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    out = landmarks_translated.copy()
    out[:, :2] = landmarks_translated[:, :2] @ rot.T
    return out


def normalize_14_9(landmarks_raw: np.ndarray, handedness: Optional[str] = None, ref_pair: Tuple[int, int] = (0, 9)) -> np.ndarray:
    """
    Normalization used in Skeleton14_9.
    """
    landmarks = landmarks_raw.copy()
    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]
        
    origin = landmarks[0]
    translated = landmarks - origin
    translated = canonicalize_rotation_14_9(translated, ref_pair)
    
    idx1, idx2 = ref_pair
    ref_distance = np.linalg.norm(translated[idx1] - translated[idx2])
    scaled = translated / ref_distance if ref_distance > 1e-6 else translated
    return scaled.astype(np.float32)


def compute_bone_vectors(landmarks: np.ndarray) -> np.ndarray:
    """
    Computes 20 anatomical bone vectors from 21 landmarks:
    Thumb: (0->1, 1->2, 2->3, 3->4)
    Index: (0->5, 5->6, 6->7, 7->8)
    Middle: (0->9, 9->10, 10->11, 11->12)
    Ring: (0->13, 13->14, 14->15, 15->16)
    Pinky: (0->17, 17->18, 18->19, 19->20)
    Returns: shape (20, 3)
    """
    bones = []
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    for chain in chains:
        for i in range(len(chain) - 1):
            bone = landmarks[chain[i + 1]] - landmarks[chain[i]]
            bones.append(bone)
    return np.array(bones, dtype=np.float32)


def compute_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Computes joint flexion angles (cosine of angle between consecutive bones)
    for all 5 fingers.
    Returns: shape (15,) values in range [-1, 1]
    """
    angles = []
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    for chain in chains:
        for i in range(len(chain) - 2):
            v1 = landmarks[chain[i + 1]] - landmarks[chain[i]]
            v2 = landmarks[chain[i + 2]] - landmarks[chain[i + 1]]
            norm1 = np.linalg.norm(v1) + 1e-7
            norm2 = np.linalg.norm(v2) + 1e-7
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angles.append(np.clip(cos_angle, -1.0, 1.0))
    return np.array(angles, dtype=np.float32)


def normalize_robust_optimized(
    landmarks_raw: np.ndarray,
    handedness: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Optimized Feature Representation:
    1. Wrist Centering: p'_i = p_i - p_0
    2. Multi-point Palm Scale Normalization:
       scale = mean(||p_5 - p_0||, ||p_9 - p_0||, ||p_17 - p_0||)
       This is far more robust against single joint tracking noise.
    3. Global Hand Orientation Vector preserved: v_orient = (p_9 - p_0) / ||p_9 - p_0||
    4. Bone Vectors (20, 3)
    5. Joint Flexion Angles (15,)
    
    Returns a dict with:
      - 'joints': shape (21, 3)
      - 'bones': shape (20, 3)
      - 'angles': shape (15,)
      - 'orientation': shape (3,)
      - 'combined_flat': shape (21*3 + 20*3 + 15 + 3 = 141,)
    """
    landmarks = landmarks_raw.copy()
    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]
        
    origin = landmarks[0]
    translated = landmarks - origin
    
    # Palm scale
    d5 = np.linalg.norm(translated[5])
    d9 = np.linalg.norm(translated[9])
    d17 = np.linalg.norm(translated[17])
    scale = (d5 + d9 + d17) / 3.0
    if scale < 1e-6:
        scale = d9 if d9 > 1e-6 else 1.0
        
    joints_scaled = (translated / scale).astype(np.float32)
    bones = compute_bone_vectors(joints_scaled)
    angles = compute_joint_angles(joints_scaled)
    
    # Global orientation unit vector
    orient_vec = translated[9] / (d9 + 1e-7)
    
    combined = np.concatenate([
        joints_scaled.flatten(),
        bones.flatten(),
        angles,
        orient_vec.astype(np.float32)
    ])
    
    return {
        'joints': joints_scaled,
        'bones': bones,
        'angles': angles,
        'orientation': orient_vec.astype(np.float32),
        'combined_flat': combined.astype(np.float32)
    }
