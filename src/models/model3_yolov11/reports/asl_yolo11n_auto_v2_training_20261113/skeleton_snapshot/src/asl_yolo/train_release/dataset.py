from pathlib import Path


def verify_release(root: Path, cfg: dict) -> dict:
    """Kiểm tra release bất biến và tập train/val đã khóa."""
    # TODO: mapping/count/fingerprint, checksum, geometry, background, leakage.
    raise NotImplementedError


def prepare_dataset(release: Path, samples: list[dict], output: Path) -> Path:
    """Tạo dataset adapter riêng, chỉ có train và validation."""
    # TODO: materialize accepted rows, portable YAML, không sửa release.
    raise NotImplementedError


def select_smoke(samples: list[dict], seed: int) -> list[dict]:
    """Chọn subset kỹ thuật cố định theo lớp và loại ảnh."""
    # TODO: stratified train/val sampling, preserve release split.
    raise NotImplementedError
