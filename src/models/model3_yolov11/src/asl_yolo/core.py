from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.request


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only a complete JSON document, preserving the previous version on failure.
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rgb(path):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def load_config(path):
    cfg = read_json(path)
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    names = cfg.get("classes", [])
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("classes must be unique nonempty strings")
    ratios = cfg.get("split", [])
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("split requires three positive ratios summing to one")
    for key in ("max_per_class", "imgsz", "batch", "smoke_epochs", "baseline_epochs"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("allow_download", "allow_training", "allow_final_test", "allow_webcam", "require_reviewed_boxes"):
        if type(cfg.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not cfg["require_reviewed_boxes"]:
        raise ValueError("Reviewed boxes are mandatory")
    if cfg.get("model") != "yolo11n.pt":
        raise ValueError("This pipeline requires pretrained yolo11n.pt")
    if type(cfg.get("seed")) is not int or type(cfg.get("freeze")) is not int or cfg["freeze"] < 0:
        raise ValueError("seed/freeze must be integers; freeze >= 0")
    cfg.setdefault("work_name", "prototype_v1")
    if Path(cfg["work_name"]).name != cfg["work_name"] or cfg["work_name"] in (".", ".."):
        raise ValueError("work_name must be a directory name")
    cfg.setdefault("near_duplicate_distance", 2)
    if not 0 <= cfg["near_duplicate_distance"] <= 8:
        raise ValueError("near_duplicate_distance must be 0..8")
    cfg.setdefault("box_padding", 0.15)
    if not 0 <= cfg["box_padding"] <= 1:
        raise ValueError("box_padding must be 0..1")
    return cfg


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("ultralytics", "torch", "torchvision", "mediapipe", "opencv-python", "numpy", "pandas", "scikit-learn", "kaggle"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def bootstrap_environment(cfg, root):
    root = Path(root).resolve()
    colab = Path("/content").is_dir() and ("COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules)
    cache = Path(cfg["colab_cache"]) if colab else root / ".cache" / "asl_work"
    work = root / "data" / "manifests" / cfg["work_name"]
    for p in (cache, work, root / "assets", root / "runs", root / "reports" / cfg["work_name"], root / "data/review" / cfg["work_name"], root / "data/processed", root / "exports"):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import torch
    env = {"root": str(root), "cache": str(cache), "work": str(work), "colab": colab,
           "report_dir": str(root / "reports" / cfg["work_name"]),
           "review_dir": str(root / "data/review" / cfg["work_name"]),
           "versions": versions(), "cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_json(Path(env["report_dir"]) / "environment.json", env)
    print(json.dumps(env, indent=2, ensure_ascii=True))
    return env


def download(url, path, allowed=False):
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    if not allowed:
        raise RuntimeError(f"Missing {path.name}; set allow_download=True to download from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temp.open("wb") as out:
            import shutil
            shutil.copyfileobj(response, out)
        if not temp.stat().st_size:
            raise RuntimeError("Empty download")
        temp.replace(path)
    except Exception as exc:
        raise RuntimeError(f"Download failed for {path.name}; check network or supply the asset manually") from exc
    finally:
        temp.unlink(missing_ok=True)
    return path
