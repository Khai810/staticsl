from pathlib import Path
from typing import Any


def load_training_config(path: Path) -> dict[str, Any]:
    """Đọc cấu hình và khóa cổng dữ liệu, model, final test."""
    # TODO: kiểm tra schema, release, asset, protocol và hyperparameters.
    raise NotImplementedError


def preflight(root: Path, cfg: dict, output: Path, require_gpu: bool = True) -> dict:
    """Xác minh dữ liệu, loader, môi trường và storage trước train."""
    # TODO: adapter -> hashes/leakage -> asset -> dependencies/CUDA -> bằng chứng.
    raise NotImplementedError
