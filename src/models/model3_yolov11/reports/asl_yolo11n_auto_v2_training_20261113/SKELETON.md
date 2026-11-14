# Skeleton trước triển khai

Lưu lúc 2026-11-13T13:28:47.175232+00:00. Các file dưới đây được tạo thật với NotImplementedError trước khi triển khai.

Input/output và luồng xem IMPLEMENTATION_PLAN.md. Bản chụp nguyên file: skeleton_snapshot/.

## src/asl_yolo/train_release/__init__.py

```python
"""Training riêng cho release auto V2; không thay API prototype."""

```

## src/asl_yolo/train_release/cli.py

```python
def main(argv: list[str] | None = None) -> int:
    """Điểm vào prepare/preflight/smoke/resume/baseline/analysis/package."""
    # TODO: arguments, explicit gates, no automatic final test.
    raise NotImplementedError

```

## src/asl_yolo/train_release/config.py

```python
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

```

## src/asl_yolo/train_release/dataset.py

```python
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

```

## src/asl_yolo/train_release/engine.py

```python
def trainer_class(context: dict) -> type:
    """Tạo adapter Ultralytics đã pin để validate và commit từng epoch."""
    # TODO: validator capture, model/EMA/optimizer/scheduler/RNG, OOM guard.
    raise NotImplementedError

```

## src/asl_yolo/train_release/metrics.py

```python
def score_predictions(samples: list[dict], predictions: dict, classes: list[str], supported: list[str], threshold: float) -> dict:
    """Chấm positive và background riêng, giữ NO_DETECTION."""
    # TODO: supported macro, weighted, coverage, Wilson, confusion, FP.
    raise NotImplementedError


def is_better(candidate: dict, previous: dict | None) -> bool:
    """Chọn macro F1 rồi Accuracy rồi epoch sớm hơn."""
    # TODO: deterministic tie break.
    raise NotImplementedError

```

## src/asl_yolo/train_release/reporting.py

```python
from pathlib import Path


def analyze_validation(root: Path, run: Path, cfg: dict) -> dict:
    """Khóa model, sweep validation rồi xuất metrics, figures và gallery."""
    # TODO: verify checkpoint -> predictions -> sweep -> inference.json -> report.
    raise NotImplementedError


def regenerate_figures(run: Path) -> None:
    """Vẽ lại từ artifact thật đã lưu, không inference hay chấm test."""
    # TODO: CSV/JSON -> PNG300/SVG/captions.
    raise NotImplementedError

```

## src/asl_yolo/train_release/runner.py

```python
from pathlib import Path


def run_training(root: Path, cfg: dict, mode: str, resume_run: Path | None = None, checkpoint: Path | None = None, stop_after: int | None = None) -> Path:
    """Chạy mới hoặc resume rõ run, chỉ khi mọi cổng đã đạt."""
    # TODO: preflight -> run lock -> train -> checkpoint verification -> status.
    raise NotImplementedError


def resume_probe(root: Path, cfg: dict) -> dict:
    """Dừng sau epoch đã lưu và resume trong tiến trình GPU mới."""
    # TODO: process1 stop1/total4 -> process2 resume -> verify optimizer/history/model.
    raise NotImplementedError

```

## src/asl_yolo/train_release/storage.py

```python
from pathlib import Path


def commit_epoch(run: Path, checkpoint: dict, metrics: dict, predictions: dict) -> dict:
    """Lưu một generation epoch hoàn chỉnh có checksum và commit marker."""
    # TODO: temp -> verify -> immutable files -> commit marker -> derived history.
    raise NotImplementedError


def verify_resume(run: Path, checkpoint: Path, cfg: dict) -> dict:
    """Kiểm tra run/checkpoint và bảo toàn lịch sử trước resume."""
    # TODO: run state, fingerprint/config/version, checksum, epoch/optimizer.
    raise NotImplementedError

```

## scripts/train_release.py

```python
"""Điểm vào CLI training release; skeleton được lưu trước triển khai."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl_yolo.train_release.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

```
