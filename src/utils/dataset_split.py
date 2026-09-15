"""
src/utils/dataset_split.py
=============================
File DÙNG CHUNG cho cả 3 model (đúng vị trí trong roadmap: "utils/dataset_split.py").
Chia dataset thành train/val/test.

Điểm quan trọng nhất của module này là tham số `group_column` (trong
configs.config.SplitConfig):
    - Model 1 (ảnh tĩnh độc lập) & Model 3 (YOLO): có thể chia ngẫu nhiên
      theo từng dòng (không có cột group_column trong DataFrame).
    - Model 2 (chuỗi cắt từ video bằng sliding window): BẮT BUỘC group theo
      session_id, vì nhiều mẫu chuỗi được cắt overlap từ CÙNG 1 video. Nếu
      chia ngẫu nhiên theo từng dòng, các mẫu gần giống hệt nhau (do
      overlap) sẽ lọt vào cả train và test -> Accuracy ảo cao (data leakage).

Cách dùng:
    from configs.config import PreprocessConfig
    from src.utils.dataset_split import split_dataset
    train_df, val_df, test_df = split_dataset(df, PreprocessConfig().split)
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from configs.config import SplitConfig


def split_dataset(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia DataFrame thành 3 tập train / val / test.

    Tham số:
        df (pd.DataFrame): bắt buộc có cột "label". Nếu dùng group-split,
            bắt buộc có thêm cột trùng tên với cfg.group_column
            (mặc định "session_id").
        cfg (SplitConfig): cấu hình chia tập, gồm:
            - test_size (float)   : tỉ lệ tập test trên TỔNG dataset.
            - val_size (float)    : tỉ lệ tập validation trên TỔNG dataset.
            - random_state (int)  : seed để tái lập kết quả.
            - stratify (bool)     : có chia đều theo tỉ lệ từng lớp hay không
                                     (chỉ áp dụng khi KHÔNG dùng group split).
            - group_column (str)  : tên cột group; nếu cột này TỒN TẠI trong
                                     df, hàm tự động chuyển sang GROUP SPLIT
                                     để chống rò rỉ dữ liệu.

    Trả về:
        (train_df, val_df, test_df) — 3 DataFrame con, reset index sạch,
        không trùng lặp dữ liệu giữa các tập.
    """
    if cfg.group_column in df.columns:
        return _group_split(df, cfg)
    return _stratified_split(df, cfg)


def _stratified_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia ngẫu nhiên có stratify theo nhãn — dùng cho dữ liệu độc lập (Model 1, Model 3)."""
    strat_col = df["label"] if cfg.stratify else None

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    trainval_idx, test_idx = next(splitter1.split(df, strat_col))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    # val_size tính lại theo tỉ lệ TRÊN TẬP TRAINVAL còn lại, để tỉ lệ val
    # TRÊN TOÀN BỘ DATASET đúng bằng cfg.val_size ban đầu.
    relative_val_size = cfg.val_size / (1.0 - cfg.test_size)
    strat_col_trainval = trainval_df["label"] if cfg.stratify else None

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=cfg.random_state)
    train_idx, val_idx = next(splitter2.split(trainval_df, strat_col_trainval))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _group_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia theo GROUP (ví dụ session_id) — BẮT BUỘC dùng cho Model 2 để đảm
    bảo toàn bộ mẫu chuỗi cắt từ CÙNG 1 video luôn nằm cùng 1 tập.
    """
    groups = df[cfg.group_column]

    splitter1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    trainval_idx, test_idx = next(splitter1.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    relative_val_size = cfg.val_size / (1.0 - cfg.test_size)
    groups_trainval = trainval_df[cfg.group_column]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=cfg.random_state)
    train_idx, val_idx = next(splitter2.split(trainval_df, groups=groups_trainval))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    _assert_no_group_leakage(train_df, val_df, test_df, cfg.group_column)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _assert_no_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str):
    """Kiểm tra an toàn: đảm bảo không có group_id nào xuất hiện ở > 1 tập."""
    train_groups = set(train_df[group_column])
    val_groups = set(val_df[group_column])
    test_groups = set(test_df[group_column])

    assert train_groups.isdisjoint(val_groups), "RÒ RỈ DỮ LIỆU: có group trùng giữa train và val!"
    assert train_groups.isdisjoint(test_groups), "RÒ RỈ DỮ LIỆU: có group trùng giữa train và test!"
    assert val_groups.isdisjoint(test_groups), "RÒ RỈ DỮ LIỆU: có group trùng giữa val và test!"
