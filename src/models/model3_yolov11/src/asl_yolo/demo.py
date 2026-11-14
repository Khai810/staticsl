from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from .core import file_hash, read_json, write_json
from .training import load_run
from .vision import HandTracker, landmarks_box
from .webcam import SingleHandLock, score_predictor_class, select_hand_scores, hand_crop, draw_dashboard, box_iou

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def overlay(frame, result, hands, classes):
    import cv2
    canvas = frame.copy()
    h, w = frame.shape[:2]
    for hand in hands:
        points = [(int(p[0] * w), int(p[1] * h)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(canvas, points[a], points[b], (255, 180, 0), 1)
        for point in points:
            cv2.circle(canvas, point, 2, (255, 255, 0), -1)
    boxes = result.boxes
    for box, cid, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{classes[int(cid)]} {confidence:.2f}", (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
    if len(boxes) == 0:
        cv2.putText(canvas, "NO_DETECTION", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    return canvas


def demo_main(argv=None):
    import cv2
    parser = argparse.ArgumentParser(description="Windows webcam / uploaded video ASL inference")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source", default="0", help="Camera index or input video path")
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--output", type=Path, help="Save processed video, especially for Colab")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--confidence", type=float, help="Webcam default .25; video defaults to saved validation threshold")
    parser.add_argument("--input-mode", choices=("full", "hand-crop"), default="full")
    parser.add_argument("--status-file", type=Path, help="Write live diagnostic scores and counters, without camera images")
    parser.add_argument("--max-frames", type=int, help="Optional bounded diagnostic run")
    args = parser.parse_args(argv)
    if args.width < 1:
        parser.error("--width must be positive")
    if args.confidence is not None and not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0,1]")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.no_display and args.output is None:
        parser.error("--no-display requires --output")
    source = int(args.source) if args.source.isdecimal() else args.source
    if args.no_display and isinstance(source, int):
        parser.error("For headless/Colab usage provide an uploaded video file")
    model, classes, inference, _ = load_run(args.run_dir)
    threshold = args.confidence if args.confidence is not None else (.25 if isinstance(source, int) else inference["confidence"])
    input_mode = args.input_mode
    lock = SingleHandLock()
    predictor_class = score_predictor_class()
    asset = args.asset or args.run_dir / "assets/hand_landmarker.task"
    if not asset.exists():
        raise FileNotFoundError(f"Hand model missing: {asset}; use --asset path or exported bundle")
    tracker, capture, writer = None, None, None
    frames, mp_failures, processing_seconds = 0, 0, 0.
    accepted_frames, locked_frames, maximum_score, last_status = 0, 0, 0., 0.
    try:
        try:
            tracker = HandTracker(asset, video=True)
        except Exception as exc:
            print(f"MediaPipe unavailable ({type(exc).__name__}); YOLO will continue without landmarks")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera/video. Check source, Windows desktop camera permission and competing apps. Colab cannot access the PC camera via index 0.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 240:
            fps = 25.
        while True:
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, int):
                    raise RuntimeError("Camera frame read failed")
                break
            start = time.perf_counter()
            if frame.shape[1] > args.width:
                frame = cv2.resize(frame, (args.width, round(frame.shape[0] * args.width / frame.shape[1])))
            hands = []
            if tracker:
                try:
                    hands = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except Exception:
                    hands = []
            if not hands:
                mp_failures += 1
            # Always run YOLO on the unannotated frame, even if MediaPipe fails.
            result = model.predict(frame, conf=.001, imgsz=inference["imgsz"], iou=inference["iou"],
                                   verbose=False, predictor=predictor_class)[0]
            h, w = frame.shape[:2]
            candidates = []
            for i, hand in enumerate(hands):
                try:
                    box = np.asarray(landmarks_box(hand, w, h, .15)) / [w, h, w, h]
                    handedness = tracker.handedness[i] if i < len(tracker.handedness) else (None, 0.)
                    candidates.append({"box": box, "landmarks": hand, "handedness": handedness, "source": "MediaPipe"})
                except ValueError:
                    continue
            if not candidates:
                # A YOLO-only hand box can keep the demo alive during MP failures.
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    if confidence >= threshold:
                        normalized = box / [w, h, w, h]
                        # Merge different class proposals at the same hand before association.
                        if any(box_iou(normalized, [c["box"]])[0] > .5 for c in candidates):
                            continue
                        candidates.append({"box": normalized, "landmarks": None, "source": "YOLO fallback"})
            target = lock.update(candidates, time.monotonic())
            state = "LOCK LOST" if lock.lost else "WAITING FOR HAND" if lock.target is None else "WAITING FOR LOCKED HAND"
            scores, predicted_box = None, None
            active_input = "full"
            if target is not None:
                state = "HAND 1 LOCKED / " + target["source"]
                locked_frames += 1
                target_box = np.asarray(target["box"]) * [w, h, w, h]
                other_boxes = [np.asarray(c["box"]) * [w, h, w, h] for c in candidates if c is not target
                               and not np.allclose(c["box"], target["box"])]
                boxes, raw_scores = model.predictor.letter_boxes, model.predictor.letter_scores
                if input_mode == "hand-crop":
                    cropped, offset = hand_crop(frame, target["box"])
                    if cropped.size:
                        model.predict(cropped, conf=.001, imgsz=inference["imgsz"], iou=inference["iou"], verbose=False)
                        boxes = model.predictor.letter_boxes + np.tile(offset, 2)
                        raw_scores = model.predictor.letter_scores
                        active_input = "hand-crop"
                scores, predicted_box = select_hand_scores(boxes, raw_scores, target_box, other_boxes)
                maximum_score = max(maximum_score, float(scores.max()))
                accepted_frames += int(predicted_box is not None and scores.max() >= threshold)
            processing_seconds += time.perf_counter() - start
            frames += 1
            canvas = draw_dashboard(frame, classes, scores, target, predicted_box, threshold,
                                    frames / processing_seconds, state, active_input)
            if args.status_file and time.monotonic() - last_status >= 1:
                write_json(args.status_file, {"frames": frames, "locked_frames": locked_frames,
                           "accepted_frames": accepted_frames, "max_score": maximum_score, "state": state,
                           "input_mode": active_input, "threshold": threshold, "saved_validation_threshold": inference["confidence"],
                           "scores": dict(zip(classes, map(float, scores))) if scores is not None else None,
                           "fps": frames / processing_seconds})
                last_status = time.monotonic()
            if args.output:
                if writer is None:
                    if args.output.exists():
                        raise FileExistsError("Output exists; choose a new file")
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Video writer unavailable")
                writer.write(canvas)
            if not args.no_display:
                try:
                    cv2.imshow("ASL YOLO11n - Q to quit", canvas)
                    key = cv2.waitKey(1) & 255
                    if frames == 1:
                        print(f"WEBCAM_READY source={source} frame={canvas.shape[1]}x{canvas.shape[0]} "
                              f"classes={len(classes)} webcam_confidence={threshold:.2f} "
                              f"saved_validation_confidence={inference['confidence']:.2f}", flush=True)
                    if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("ASL YOLO11n - Q to quit", cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord("r"), ord("R")):
                        lock.reset()
                    elif key in (ord("c"), ord("C")):
                        input_mode = "hand-crop" if input_mode == "full" else "full"
                    elif key in (ord("["), ord("]")):
                        threshold = round(min(.95, max(.05, threshold + (.05 if key == ord("]") else -.05))), 2)
                except cv2.error as exc:
                    raise RuntimeError("OpenCV GUI unavailable. For Colab use --source video.mp4 --no-display --output result.mp4") from exc
            if args.max_frames is not None and frames >= args.max_frames:
                break
        if not frames:
            raise RuntimeError("No decoded frames")
        print({"frames": frames, "mediapipe_failures": mp_failures, "locked_frames": locked_frames,
               "accepted_frames": accepted_frames, "max_score": maximum_score,
               "processing_fps_excluding_capture_display": frames / processing_seconds}, flush=True)
    finally:
        if writer:
            writer.release()
        if capture:
            capture.release()
        if tracker:
            tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def export_demo_bundle(run_dir, env):
    run = Path(run_dir)
    _, _, inference, checkpoint = load_run(run)
    root = Path(env["root"])
    destination = root / "exports" / run.name
    if destination.exists():
        raise FileExistsError("Bundle already exists; choose a new destination/run")
    asset = root / "assets/hand_landmarker.task"
    if not asset.exists():
        raise FileNotFoundError("MediaPipe asset required for demo bundle")
    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / "model.pt")
    for name in ("class_mapping.json", "inference.json", "config.json", "versions.json", "manifest.json"):
        shutil.copy2(run / name, destination / name)
    shutil.copy2(root / "requirements-local.txt", destination / "requirements-local.txt")
    shutil.copytree(root / "src/asl_yolo", destination / "src/asl_yolo", ignore=shutil.ignore_patterns("__pycache__"))
    (destination / "scripts").mkdir()
    shutil.copy2(root / "scripts/webcam_demo.py", destination / "scripts/webcam_demo.py")
    (destination / "assets").mkdir()
    shutil.copy2(asset, destination / "assets/hand_landmarker.task")
    (destination / "README.txt").write_text("Windows Python 3.11: create .venv, install requirements-local.txt, then run:\n.venv\\Scripts\\python.exe scripts/webcam_demo.py --run-dir .\nQ exits. MediaPipe failure does not block YOLO.\n", encoding="utf-8")
    write_json(destination / "checksums.json", {p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob("*") if p.is_file()})
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in destination.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(destination))
    return archive
