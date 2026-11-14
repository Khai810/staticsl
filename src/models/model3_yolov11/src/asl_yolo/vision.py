from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .core import download, file_hash, read_json, read_rgb, write_json

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def normalize_landmarks(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite (21,3) landmarks")
    shifted = points - points[0]
    scale = np.linalg.norm(shifted[[5, 9, 13, 17]], axis=1).mean()
    if scale < 1e-8:
        raise ValueError("Degenerate palm scale")
    return (shifted / scale).reshape(63).tolist()


def clamp_box(box, width, height):
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("Invalid xyxy box")
    x1, y1, x2, y2 = box
    box = [max(0., min(float(width), x1)), max(0., min(float(height), y1)),
           max(0., min(float(width), x2)), max(0., min(float(height), y2))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box has zero/negative area")
    return box


def landmarks_box(points, width, height, padding=.15):
    xy = np.asarray(points, dtype=float)[:, :2] * [width, height]
    low, high = xy.min(axis=0), xy.max(axis=0)
    pad = (high - low) * padding
    return clamp_box([*(low - pad), *(high + pad)], width, height)


def xyxy_to_yolo(box, width, height):
    x1, y1, x2, y2 = clamp_box(box, width, height)
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]


def yolo_to_xyxy(box, width, height):
    x, y, w, h = box
    return [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]


class HandTracker:
    def __init__(self, asset, video=False):
        import mediapipe as mp
        self.mp = mp
        self.video = video
        self.last_timestamp = -1
        self.handedness = []
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)), num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            stamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1_000_000)
            self.last_timestamp = stamp
            result = self.detector.detect_for_video(image, stamp)
        else:
            result = self.detector.detect(image)
        self.handedness = [(categories[0].category_name, categories[0].score) if categories else (None, 0.)
                           for categories in result.handedness]
        return [[[p.x, p.y, p.z] for p in hand] for hand in result.hand_landmarks]

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def extract_landmarks(manifest, dataset, cfg, env):
    asset = download(HAND_MODEL_URL, Path(env["root"]) / "assets/hand_landmarker.task", cfg["allow_download"])
    signature = {"manifest": manifest["fingerprint"], "asset": file_hash(asset), "padding": cfg["box_padding"]}
    path = Path(env["work"]) / "landmarks.json"
    if path.exists():
        saved = read_json(path)
        if saved["signature"] != signature:
            raise ValueError("Landmark cache differs; use new work_name")
        return saved
    items = {}
    with HandTracker(asset) as tracker:
        for row in manifest["rows"]:
            flags, boxes, vectors, hands = [], [], [], []
            try:
                hands = tracker.detect(read_rgb(Path(dataset["extracted"]) / row["path"]))
                if not hands:
                    flags.append("mp_no_hand")
                if len(hands) > 1:
                    flags.append("multiple_hands")
                for points in hands:
                    vectors.append(normalize_landmarks(points))
                    boxes.append(landmarks_box(points, row["width"], row["height"], cfg["box_padding"]))
                    xy = np.asarray(points)[:, :2]
                    if (xy < .01).any() or (xy > .99).any():
                        flags.append("possible_out_of_frame")
            except Exception as exc:
                flags.append("mp_error:" + type(exc).__name__)
            if row["label_conflict"]:
                flags.append("duplicate_label_conflict")
            items[row["sample_id"]] = {"landmarks": hands, "vectors63": vectors, "boxes": boxes,
                                       "flags": sorted(set(flags)), "mp_failed": not hands or any(f.startswith("mp_error") for f in flags)}
    result = {"signature": signature, "items": items}
    write_json(path, result)
    report = {}
    for label in manifest["classes"]:
        group = [items[r["sample_id"]] for r in manifest["rows"] if r["class_name"] == label]
        report[label] = {"total": len(group), "failures": sum(r["mp_failed"] for r in group)}
    write_json(Path(env["report_dir"]) / "mediapipe_failures.json", report)
    return result
