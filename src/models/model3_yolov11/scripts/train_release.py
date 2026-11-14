"""Điểm vào CLI training release; skeleton được lưu trước triển khai."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asl_yolo.train_release.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
