from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import gc
import shutil
import uuid
from .core import digest, download, file_hash, read_json, versions, write_json
from .export import validate_yolo
from .review import require_reviewed


def assert_mapping(actual, expected):
    if isinstance(actual, dict):
        actual = [actual[k] for k in sorted(actual, key=int)]
    if list(actual) != list(expected):
        raise ValueError(f"Checkpoint class mapping mismatch: {actual} != {expected}")


def run_checkpoint(run_dir):
    run_dir = Path(run_dir)
    state = read_json(run_dir / "run.json")
    checkpoint = run_dir / state["best"]
    if not checkpoint.is_file():
        raise FileNotFoundError("Run has no completed best.pt")
    return checkpoint


def train_detector(artifacts, manifest, review_path, cfg, env, mode="smoke", resume_run=None, allow_cpu=False):
    if not cfg["allow_training"]:
        raise RuntimeError("Training disabled. Enable allow_training only after review and validation.")
    if mode not in ("smoke", "baseline"):
        raise ValueError("mode must be smoke or baseline")
    import torch
    from ultralytics import YOLO
    review = require_reviewed(manifest, review_path)
    if artifacts["signature"] != {"manifest": manifest["fingerprint"], "review": digest(review)}:
        raise ValueError("Export snapshot does not match current manifest/review")
    validate_yolo(artifacts["cache"], manifest, review)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("No CUDA GPU. Use Colab GPU/T4; CPU training requires explicit allow_cpu=True.")
    root = Path(env["root"])
    settings = {k: cfg[k] for k in ("model", "imgsz", "batch", "smoke_epochs", "baseline_epochs", "freeze", "seed", "classes")}
    signature = {"dataset": artifacts["signature"], "settings": settings, "mode": mode}
    if resume_run:
        run = Path(resume_run)
        state = read_json(run / "run.json")
        if state["signature"] != signature:
            raise ValueError("Resume dataset/config mismatch; use a separate new experiment")
        if state.get("status") == "completed":
            raise ValueError("Completed run cannot resume; warm start is a separate experiment")
        checkpoint = run / state["last"]
        if not checkpoint.exists():
            raise FileNotFoundError("No last.pt yet: at least one epoch must finish before resume")
        model = YOLO(str(checkpoint))
        assert_mapping(model.names, manifest["classes"])
        def resume_saved(trainer):
            write_json(run / "progress.json", {"epoch": int(trainer.epoch) + 1, "save_dir": str(trainer.save_dir), "resumed": True})
        model.add_callback("on_model_save", resume_saved)
        model.train(resume=True, data=artifacts["yaml"], save_dir=str(checkpoint.parent.parent),
                    device=0 if torch.cuda.is_available() else "cpu")
        assert_mapping(model.names, manifest["classes"])
        state["status"] = "completed"
        write_json(run / "run.json", state)
        return str(run)
    pretrained = download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", root / "assets/yolo11n.pt", cfg["allow_download"])
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + mode + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", cfg)
    write_json(run / "class_mapping.json", manifest["classes"])
    write_json(run / "manifest.json", manifest)
    write_json(run / "review.json", review)
    write_json(run / "environment.json", env)
    write_json(run / "versions.json", versions())
    shutil.copy2(root / "requirements-local.txt", run / "requirements-local.txt")
    state = {"signature": signature, "status": "starting", "pretrained_sha256": file_hash(pretrained), "attempts": []}
    batch = cfg["batch"]
    while True:
        attempt = f"attempt_{len(state['attempts']) + 1:02d}"
        model = YOLO(str(pretrained))
        backbone_count = len(model.model.yaml["backbone"])
        if cfg["freeze"] > backbone_count:
            raise ValueError(f"freeze must be <= actual backbone count {backbone_count}")
        layers = [{"index": i, "type": type(layer).__name__, "frozen": i < cfg["freeze"]} for i, layer in enumerate(model.model.model)]
        write_json(run / "layers.json", {"backbone_layers": backbone_count, "layers": layers})
        state.update(status="running", best=f"{attempt}/weights/best.pt", last=f"{attempt}/weights/last.pt")
        state["attempts"].append({"name": attempt, "batch": batch})
        write_json(run / "run.json", state)
        def epoch_saved(trainer):
            # Trainer save_dir is already persistent Drive storage, including every epoch's last.pt.
            write_json(run / "progress.json", {"epoch": int(trainer.epoch) + 1, "save_dir": str(trainer.save_dir), "batch": batch})
        model.add_callback("on_model_save", epoch_saved)
        try:
            model.train(data=artifacts["yaml"], epochs=cfg["smoke_epochs"] if mode == "smoke" else cfg["baseline_epochs"],
                imgsz=cfg["imgsz"], batch=batch, device=0 if torch.cuda.is_available() else "cpu", workers=2 if env["colab"] else 0,
                project=str(run), name=attempt, exist_ok=False, save=True, save_period=1, seed=cfg["seed"], deterministic=True,
                freeze=cfg["freeze"], fliplr=0., flipud=0., mosaic=0., mixup=0., degrees=0., shear=0., perspective=0.,
                translate=.05, scale=.1, copy_paste=0., amp=False, plots=True, patience=cfg["baseline_epochs"], cache=False)
            assert_mapping(model.names, manifest["classes"])
            state["status"] = "completed"
            write_json(run / "run.json", state)
            return str(run)
        except RuntimeError as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower()
            state["status"] = "oom" if oom else "failed"
            state["attempts"][-1]["error"] = str(exc)
            write_json(run / "run.json", state)
            if not oom or batch <= 1:
                raise
            batch = max(1, batch // 2)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"CUDA OOM: restarting a NEW attempt from pretrained weights at batch={batch}. Prior attempt retained.")


def load_run(run_dir, require_inference=True):
    from ultralytics import YOLO
    run = Path(run_dir)
    names = read_json(run / "class_mapping.json")
    checkpoint = run / "model.pt" if (run / "model.pt").is_file() else run_checkpoint(run)
    model = YOLO(str(checkpoint))
    assert_mapping(model.names, names)
    inference = read_json(run / "inference.json") if require_inference else None
    if inference:
        assert_mapping(inference["classes"], names)
        if inference["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Inference config belongs to a different checkpoint")
    return model, names, inference, checkpoint
