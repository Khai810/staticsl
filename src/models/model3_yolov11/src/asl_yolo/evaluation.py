from __future__ import annotations
from collections import Counter
from pathlib import Path
import time
import numpy as np
from .core import digest, file_hash, read_json, read_rgb, write_json
from .training import load_run
from .export import validate_yolo
from .vision import HandTracker, normalize_landmarks

NO_DETECTION = "NO_DETECTION"


def label_metrics(truth, predicted, classes):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    labels = list(classes) + [NO_DETECTION]
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Need equal, nonempty truth/prediction lists")
    p, r, f, s = precision_recall_fscore_support(truth, predicted, labels=classes, zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(np.mean(f)),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)},
            "confusion_labels": labels, "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
            "no_detection_count": predicted.count(NO_DETECTION), "n": len(truth)}


def collect_predictions(model, rows, cache, imgsz, confidence=.001):
    predictions = {}
    for row in rows:
        path = Path(cache) / "images" / row["split"] / (row["sample_id"] + ".png")
        result = model.predict(str(path), imgsz=imgsz, conf=confidence, iou=.7, verbose=False)[0]
        boxes = result.boxes
        predictions[row["sample_id"]] = [{"class_id": int(cid), "confidence": float(conf), "box": list(map(float, box))}
            for cid, conf, box in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist())]
    return predictions


def score_predictions(rows, predictions, review, classes, confidence):
    truth, predicted, negatives = [], [], []
    multi = 0
    for row in rows:
        candidates = [p for p in predictions[row["sample_id"]] if p["confidence"] >= confidence]
        multi += len(candidates) > 1
        label = classes[max(candidates, key=lambda p: p["confidence"])["class_id"]] if candidates else NO_DETECTION
        if review["items"][row["sample_id"]]["state"] == "negative":
            negatives.append(label == NO_DETECTION)
        else:
            truth.append(row["class_name"])
            predicted.append(label)
    metrics = label_metrics(truth, predicted, classes)
    metrics.update(total_images=len(rows), multiple_detection_images=multi, confirmed_negative_images=len(negatives),
                   negative_correct_rejections=sum(negatives),
                   policy="Highest-confidence box per positive image; no detection is wrong. Confirmed no-hand negatives reported separately; all images included in detection mAP.")
    return metrics


def select_inference_config(run_dir, artifacts, cfg):
    run = Path(run_dir)
    model, classes, _, checkpoint = load_run(run, False)
    manifest, review = read_json(run / "manifest.json"), read_json(run / "review.json")
    if read_json(run / "run.json")["signature"]["dataset"] != artifacts["signature"]:
        raise ValueError("Dataset/run mismatch")
    validate_yolo(artifacts["cache"], manifest, review)
    target = run / "inference.json"
    if target.exists():
        saved = read_json(target)
        if saved["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Frozen threshold belongs to a different checkpoint")
        return saved
    if (run / "test_evaluation.json").exists():
        raise ValueError("Cannot tune threshold after final test")
    rows = [r for r in manifest["rows"] if r["split"] == "val"]
    predictions = collect_predictions(model, rows, artifacts["cache"], cfg["imgsz"], .01)
    scores = [{"confidence": float(t), **score_predictions(rows, predictions, review, classes, float(t))} for t in np.arange(.05, .96, .05)]
    best = max(scores, key=lambda s: (s["macro_f1"], s["accuracy"], s["confidence"]))
    inference = {"confidence": best["confidence"], "imgsz": cfg["imgsz"], "iou": .7, "classes": classes,
                 "checkpoint_sha256": file_hash(checkpoint), "manifest": manifest["fingerprint"],
                 "selected_on": "val", "criterion": "macro_f1, then accuracy, then higher threshold", "rule": "highest_confidence_box"}
    write_json(run / "validation_sweep.json", scores)
    write_json(run / "validation_predictions.json", predictions)
    write_json(target, inference)
    return inference


def evaluate_test(run_dir, artifacts, cfg, landmarks=None):
    if not cfg["allow_final_test"]:
        raise RuntimeError("Final test disabled; freeze validation decisions first")
    run = Path(run_dir)
    model, classes, inference, _ = load_run(run)
    manifest, review = read_json(run / "manifest.json"), read_json(run / "review.json")
    if inference["selected_on"] != "val" or inference["manifest"] != manifest["fingerprint"]:
        raise ValueError("Threshold must come from validation of this run")
    if read_json(run / "run.json")["signature"]["dataset"] != artifacts["signature"]:
        raise ValueError("Dataset/run mismatch")
    validate_yolo(artifacts["cache"], manifest, review)
    target = run / "test_evaluation.json"
    signature = digest({"inference": inference, "dataset": artifacts["signature"]})
    if target.exists():
        saved = read_json(target)
        if saved["signature"] != signature:
            raise ValueError("Final test already evaluated under another configuration")
        return saved
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    if landmarks and landmarks["signature"]["manifest"] != manifest["fingerprint"]:
        raise ValueError("MediaPipe report belongs to a different manifest")
    predictions = collect_predictions(model, rows, artifacts["cache"], inference["imgsz"], .001)
    metrics = score_predictions(rows, predictions, review, classes, inference["confidence"])
    result = model.val(data=artifacts["yaml"], split="test", imgsz=inference["imgsz"], conf=.001,
                       iou=inference["iou"], plots=True, project=str(run), name="test_detection", exist_ok=False, workers=0)
    metrics.update(mAP50=float(result.box.map50), mAP50_95=float(result.box.map),
                   mediapipe_failure_rate=(sum(landmarks["items"][r["sample_id"]]["mp_failed"] for r in rows) / len(rows)) if landmarks else None)
    evaluation = {"signature": signature, "metrics": metrics, "predictions": predictions, "split": "test"}
    write_json(target, evaluation)
    plot_confusion(metrics, run / "label_confusion_matrix.png")
    return evaluation


def plot_confusion(metrics, path):
    import matplotlib.pyplot as plt
    labels = metrics["confusion_labels"]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Ground truth", title="Label confusion matrix (NO_DETECTION included)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_errors(run_dir, artifacts, split="val"):
    from PIL import Image, ImageDraw
    run = Path(run_dir)
    model, classes, inference, _ = load_run(run)
    manifest, review = read_json(run / "manifest.json"), read_json(run / "review.json")
    rows = [r for r in manifest["rows"] if r["split"] == split]
    if split == "test":
        predictions = read_json(run / "test_evaluation.json")["predictions"]
    elif split == "val":
        predictions = read_json(run / "validation_predictions.json")
    else:
        raise ValueError("Analyze validation or already evaluated test only")
    output = run / f"errors_{split}"
    output.mkdir(exist_ok=True)
    pairs, records = Counter(), []
    for row in rows:
        ps = [p for p in predictions[row["sample_id"]] if p["confidence"] >= inference["confidence"]]
        label = classes[max(ps, key=lambda p: p["confidence"])["class_id"]] if ps else NO_DETECTION
        expected = row["class_name"] if review["items"][row["sample_id"]]["state"] == "approved" else NO_DETECTION
        if label == expected:
            continue
        pairs[(expected, label)] += 1
        im = Image.fromarray(read_rgb(Path(artifacts["cache"]) / "images" / split / (row["sample_id"] + ".png")))
        draw = ImageDraw.Draw(im)
        for box in review["items"][row["sample_id"]]["boxes"]:
            draw.rectangle(box, outline="lime", width=2)
            draw.text(tuple(box[:2]), "GT:" + expected, fill="lime")
        for p in ps:
            draw.rectangle(p["box"], outline="red", width=2)
            draw.text(tuple(p["box"][:2]), f"{classes[p['class_id']]} {p['confidence']:.2f}", fill="red")
        im.save(output / (row["sample_id"] + ".png"))
        records.append({"sample_id": row["sample_id"], "truth": expected, "predicted": label,
                        "hypothesis": "Chưa kết luận; kiểm tra ánh sáng, góc tay, nền và chất lượng box trên validation.",
                        "next_check": "Thu ảnh mới vào tập phân tích riêng, không chỉnh GT theo dự đoán test."})
    result = {"split": split, "top_pairs": [{"truth": a, "predicted": b, "count": n} for (a, b), n in pairs.most_common(20)], "errors": records}
    write_json(output / "analysis.json", result)
    return result


def benchmark(run_dir, artifacts, asset, repetitions=30, warmup=5):
    import cv2
    import torch
    from .demo import overlay
    model, classes, inference, _ = load_run(run_dir)
    manifest = read_json(Path(run_dir) / "manifest.json")
    rows = [r for r in manifest["rows"] if r["split"] == "val"][:10]
    frames = [cv2.cvtColor(read_rgb(Path(artifacts["cache"]) / "images/val" / (r["sample_id"] + ".png")), cv2.COLOR_RGB2BGR) for r in rows]
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times, failures = {"yolo_only_ms": [], "pipeline_ms": []}, 0
    with HandTracker(asset, video=True) as tracker:
        for i in range(warmup + repetitions):
            frame = frames[i % len(frames)]
            sync()
            start = time.perf_counter()
            model.predict(frame, conf=inference["confidence"], iou=inference["iou"], imgsz=inference["imgsz"], verbose=False)
            sync()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= warmup:
                times["yolo_only_ms"].append(elapsed)
            sync()
            start = time.perf_counter()
            try:
                hands = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                vectors = [normalize_landmarks(p) for p in hands]
            except Exception:
                hands, vectors = [], []
                if i >= warmup:
                    failures += 1
            result = model.predict(frame, conf=inference["confidence"], iou=inference["iou"], imgsz=inference["imgsz"], verbose=False)[0]
            overlay(frame, result, hands, classes)
            sync()
            if i >= warmup:
                times["pipeline_ms"].append((time.perf_counter() - start) * 1000)
    report = {key: {"mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)), "fps": float(1000 / np.mean(values))} for key, values in times.items()}
    report.update(device=str(next(model.model.parameters()).device), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  warmup=warmup, repetitions=repetitions, mediapipe_exceptions=failures,
                  scope="Preloaded validation frames; pipeline includes RGB conversion, MediaPipe, vector, YOLO, overlay. Excludes camera IO/display. Not webcam FPS or cross-device estimate.")
    write_json(Path(run_dir) / "benchmark.json", report)
    return report
