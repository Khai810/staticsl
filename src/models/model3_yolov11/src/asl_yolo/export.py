from __future__ import annotations
from pathlib import Path
import shutil
import uuid
import numpy as np
from PIL import Image
import yaml
from .core import digest, file_hash, read_json, read_rgb, write_json
from .review import require_reviewed
from .vision import xyxy_to_yolo, yolo_to_xyxy, clamp_box


def validate_yolo(directory, manifest, review):
    directory = Path(directory)
    checksums_path = directory / "file_checksums.json"
    if checksums_path.exists():
        checksums = read_json(checksums_path)
        for name, expected in checksums.items():
            path = directory / name
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"Exported data changed: {name}")
    mapping = manifest["classes"]
    all_hashes, all_groups, all_ids = {}, {}, set()
    counts = {}
    for split in ("train", "val", "test"):
        rows = [r for r in manifest["rows"] if r["split"] == split]
        ids = {r["sample_id"] for r in rows}
        images = {p.stem for p in (directory / "images" / split).glob("*.png")}
        labels = {p.stem for p in (directory / "labels" / split).glob("*.txt")}
        if ids != images or ids != labels or ids & all_ids:
            raise ValueError("Image/label/manifest one-to-one mismatch")
        all_ids.update(ids)
        for row in rows:
            for key, seen in (("sha256", all_hashes), ("split_group", all_groups)):
                value = row[key]
                if value in seen and seen[value] != split:
                    raise ValueError(f"Split leakage: {key}")
                seen[value] = split
            rgb = read_rgb(directory / "images" / split / (row["sample_id"] + ".png"))
            if rgb.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Image dimensions changed")
            lines = (directory / "labels" / split / (row["sample_id"] + ".txt")).read_text().splitlines()
            expected = review["items"][row["sample_id"]]
            if len(lines) != len(expected["boxes"]):
                raise ValueError("Annotation count mismatch")
            for line, gt in zip(lines, expected["boxes"]):
                values = list(map(float, line.split()))
                if len(values) != 5 or not np.isfinite(values).all():
                    raise ValueError("Bad YOLO label")
                cid, *box = values
                if not cid.is_integer() or not 0 <= cid < len(mapping) or int(cid) != row["class_id"]:
                    raise ValueError("Class ID mismatch")
                if not all(0 <= v <= 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                    raise ValueError("Invalid normalized geometry")
                restored = yolo_to_xyxy(box, row["width"], row["height"])
                if not np.allclose(restored, clamp_box(restored, row["width"], row["height"]), atol=1e-4) or not np.allclose(restored, gt, atol=1e-3):
                    raise ValueError("Box geometry differs from reviewed GT")
        counts[split] = len(rows)
    if any(v == 0 for v in counts.values()):
        raise ValueError("Every split must be nonempty")
    return counts


def export_yolo_dataset(manifest, dataset, review_path, cfg, env):
    review = require_reviewed(manifest, review_path)
    signature = {"manifest": manifest["fingerprint"], "review": digest(review)}
    destination = Path(env["root"]) / "data/processed" / cfg["work_name"]
    if destination.exists():
        if not (destination / "signature.json").exists() or read_json(destination / "signature.json") != signature:
            raise ValueError("Processed dataset differs. Use new work_name; never overwrite reviewed datasets.")
    else:
        temp = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex[:8])
        for split in ("train", "val", "test"):
            (temp / "images" / split).mkdir(parents=True)
            (temp / "labels" / split).mkdir(parents=True)
        for row in manifest["rows"]:
            source = Path(dataset["extracted"]) / row["path"]
            if file_hash(source) != row["sha256"]:
                raise ValueError("Raw image changed after audit")
            Image.fromarray(read_rgb(source)).save(temp / "images" / row["split"] / (row["sample_id"] + ".png"))
            lines = [str(row["class_id"]) + " " + " ".join(f"{v:.9f}" for v in xyxy_to_yolo(b, row["width"], row["height"])) for b in review["items"][row["sample_id"]]["boxes"]]
            (temp / "labels" / row["split"] / (row["sample_id"] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        write_json(temp / "file_checksums.json", {p.relative_to(temp).as_posix(): file_hash(p)
            for folder in (temp / "images", temp / "labels") for p in folder.rglob("*") if p.is_file()})
        validate_yolo(temp, manifest, review)
        write_json(temp / "signature.json", signature)
        write_json(temp / "manifest.json", manifest)
        write_json(temp / "review.json", review)
        temp.rename(destination)
    counts = validate_yolo(destination, manifest, review)
    # Colab always reads images from local cache; durable annotations stay on Drive.
    cache = Path(env["cache"]) / "yolo" / digest(signature)[:16]
    if not cache.exists():
        shutil.copytree(destination, cache)
    validate_yolo(cache, manifest, review)
    spec = {"path": str(cache.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(manifest["classes"]))}
    yaml_path = cache / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    durable_spec = {**spec, "path": str(destination.resolve())}
    (destination / "data.yaml").write_text(yaml.safe_dump(durable_spec, sort_keys=False), encoding="utf-8")
    result = {"yaml": str(yaml_path), "cache": str(cache), "persistent": str(destination), "signature": signature, "counts": counts}
    write_json(Path(env["report_dir"]) / "dataset_validation.json", result)
    return result
