"""
Skeleton Data Augmentation module:
Applies light, physiologically plausible augmentations for hand landmarks:
- Light planar rotation (within +/- 12 degrees)
- Light uniform scaling (0.93 to 1.07)
- Gaussian landmark jittering (sigma = 0.008)
- Subtle joint dropout/noise to simulate occlusion
"""

import numpy as np
from typing import Tuple, Optional

def augment_landmarks_light(
    landmarks: np.ndarray,
    rotation_range_deg: float = 12.0,
    scale_range: Tuple[float, float] = (0.93, 1.07),
    jitter_std: float = 0.008,
    joint_dropout_prob: float = 0.03,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Applies light augmentation on normalized landmarks of shape (21, 3).
    Preserves anatomical consistency while improving model generalization.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    aug = landmarks.copy()
    
    # 1. Light random rotation in XY plane
    if rotation_range_deg > 0:
        angle_deg = rng.uniform(-rotation_range_deg, rotation_range_deg)
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=np.float32)
        aug[:, :2] = aug[:, :2] @ rot_matrix.T
        
    # 2. Light random scaling
    if scale_range is not None:
        scale_factor = rng.uniform(scale_range[0], scale_range[1])
        aug = aug * scale_factor
        
    # 3. Gaussian landmark jitter (MediaPipe detector noise)
    if jitter_std > 0:
        noise = rng.normal(0.0, jitter_std, aug.shape).astype(np.float32)
        # Keep wrist (joint 0) fixed at (0,0,0) or very small jitter
        noise[0] = noise[0] * 0.2
        aug += noise
        
    # 4. Joint occlusion / dropout perturbation
    if joint_dropout_prob > 0:
        # Non-wrist joints (1..20)
        mask = rng.uniform(0.0, 1.0, size=(20,)) < joint_dropout_prob
        for idx_offset, dropped in enumerate(mask):
            if dropped:
                joint_idx = idx_offset + 1
                # Perturb joint by slight displacement
                perturb = rng.normal(0.0, jitter_std * 2.5, size=(3,)).astype(np.float32)
                aug[joint_idx] += perturb
                
    return aug.astype(np.float32)
