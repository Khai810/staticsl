from pathlib import Path


def commit_epoch(run: Path, checkpoint: dict, metrics: dict, predictions: dict) -> dict:
    """Lưu một generation epoch hoàn chỉnh có checksum và commit marker."""
    # TODO: temp -> verify -> immutable files -> commit marker -> derived history.
    raise NotImplementedError


def verify_resume(run: Path, checkpoint: Path, cfg: dict) -> dict:
    """Kiểm tra run/checkpoint và bảo toàn lịch sử trước resume."""
    # TODO: run state, fingerprint/config/version, checksum, epoch/optimizer.
    raise NotImplementedError
