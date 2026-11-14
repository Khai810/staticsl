"""Manual ASL capture with 29 labels, background subtypes and held-out people."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
import uuid

from .core import read_json, write_json
from .protocol import (
    CLASSES_29, COLLECTION_ROLES, annotation_spec, load_protocol, protocol_fingerprint,
)

FIELDS = [
    "image", "sha256", "person", "role", "condition", "label", "capture_type",
    "background", "lighting", "device", "camera_index", "session_id", "burst_id",
    "frame_index", "timestamp_utc", "elapsed_seconds", "interval_seconds",
    "width", "height", "mirrored", "protocol_sha256", "review_status",
]
TARGETS = [(label, "known_hand") for label in CLASSES_29 if label != "nothing"] + [
    ("nothing", "no_hand"), ("nothing", "other_hand"),
]
SPECIAL_KEYS = {
    ord("1"): ("del", "known_hand"), ord("2"): ("space", "known_hand"),
    ord("3"): ("nothing", "no_hand"), ord("4"): ("nothing", "other_hand"),
}


def component(value):
    value = value.strip()
    if (not value or value in {".", ".."} or len(value) > 40 or
            re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or value.endswith((".", " "))):
        raise argparse.ArgumentTypeError("Use a short, valid folder name")
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)]}
    if value.split(".")[0].upper() in reserved:
        raise argparse.ArgumentTypeError("This folder name is reserved by Windows")
    return value


def claim_person_role(roles, person, role):
    if role not in COLLECTION_ROLES:
        raise ValueError("Unknown collection role")
    key = person.casefold()
    previous = roles.get(key)
    if previous is not None and previous != role:
        raise ValueError(f"Person {person} already belongs to {previous}; do not mix with {role}")
    return {**roles, key: role}


def target_for_key(key):
    if key in SPECIAL_KEYS:
        return SPECIAL_KEYS[key]
    if 0 <= key < 256 and chr(key).upper() in CLASSES_29[:26]:
        return chr(key).upper(), "known_hand"
    return None


def run(args):
    protocol = load_protocol(args.protocol)
    fingerprint = protocol_fingerprint(protocol)
    root = Path(args.output).resolve()
    if args.frames < 1 or args.interval < .25:
        raise ValueError("frames >= 1 and interval >= 0.25 seconds")
    if args.dry_run:
        print({"output": str(root), "person": args.person, "role": args.role,
               "classes": len(CLASSES_29), "capture_targets": len(TARGETS),
               "camera_opened": False, "protocol_sha256": fingerprint})
        return 0

    import cv2
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".capture.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    camera = None
    saved = 0
    try:
        snapshot = root / "protocol.json"
        metadata = root / "metadata.csv"
        if snapshot.exists():
            if protocol_fingerprint(load_protocol(snapshot)) != fingerprint:
                raise ValueError("Capture protocol changed; use a separate output directory")
        elif metadata.exists():
            raise ValueError("Existing metadata lacks its protocol snapshot; use a new output")
        else:
            write_json(snapshot, protocol)
        if metadata.exists():
            with metadata.open(encoding="utf-8-sig", newline="") as stream:
                if next(csv.reader(stream), []) != FIELDS:
                    raise ValueError("metadata.csv schema differs; use a new output")
        roles_file = root / "person_roles.json"
        roles = claim_person_role(read_json(roles_file) if roles_file.exists() else {}, args.person, args.role)
        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")
        write_json(roles_file, roles)
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        write_json(root / "sessions" / (session + ".json"), {
            "arguments": vars(args), "protocol_sha256": fingerprint,
            "review_status": "pending_review", "mirrored": False,
        })
        target = ("A", "known_hand")
        burst = None
        index = 0
        next_frame = 0.
        burst_start = 0.
        previous = None
        print("A-Z: select sign; 1: del; 2: space; 3: no hand; 4: other hand.")
        print("ENTER: capture; SPACE: cancel burst; ESC: exit. Q is the letter Q.")
        with metadata.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
                stream.flush()
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                now = time.monotonic()
                if burst is not None and now >= next_frame:
                    label, capture_type = target
                    annotation_spec(label, capture_type)
                    folder = root / args.role / args.person / session / (label + "_" + capture_type)
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{burst}_{index:03d}.png"
                    ok, encoded = cv2.imencode(".png", frame)
                    if not ok:
                        raise RuntimeError("Cannot encode PNG")
                    content = encoded.tobytes()
                    with path.open("xb") as image_file:
                        image_file.write(content)
                    writer.writerow({
                        "image": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "person": args.person, "role": args.role, "condition": args.condition,
                        "label": label, "capture_type": capture_type,
                        "background": args.background, "lighting": args.lighting,
                        "device": args.device, "camera_index": args.camera,
                        "session_id": session, "burst_id": burst, "frame_index": index,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": now - burst_start,
                        "interval_seconds": "" if previous is None else now - previous,
                        "width": frame.shape[1], "height": frame.shape[0], "mirrored": False,
                        "protocol_sha256": fingerprint, "review_status": "pending_review",
                    })
                    stream.flush()
                    previous = now
                    index += 1
                    saved += 1
                    next_frame = now + args.interval
                    if index >= args.frames:
                        burst = None
                preview = frame.copy()
                status = f"{target[0]} / {target[1]} | Saved {saved} | ENTER capture | ESC exit"
                cv2.putText(preview, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
                cv2.imshow("ASL29 capture - unmirrored", preview)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                if key == 32:
                    burst = None
                if burst is not None:
                    continue
                selected = target_for_key(key)
                if selected is not None:
                    target = selected
                elif key in (10, 13):
                    burst = uuid.uuid4().hex[:10]
                    index = 0
                    burst_start = time.monotonic()
                    next_frame = burst_start
                    previous = None
        print(f"Saved {saved} images. Metadata: {metadata}")
        return saved
    finally:
        try:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
        finally:
            lock.unlink()


def audit_capture(root, min_per_target=1):
    """Audit collection integrity/coverage; this does not approve labels or training."""
    if min_per_target < 1:
        raise ValueError("min_per_target must be positive")
    root = Path(root).resolve()
    protocol = load_protocol(root / "protocol.json")
    fingerprint = protocol_fingerprint(protocol)
    errors = []
    coverage = defaultdict(Counter)
    conditions = defaultdict(set)
    roles = {}
    hashes = defaultdict(list)
    images = set()
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Capture metadata schema differs")
        for number, row in enumerate(reader, 2):
            try:
                annotation_spec(row["label"], row["capture_type"])
                roles = claim_person_role(roles, row["person"], row["role"])
                if row["protocol_sha256"] != fingerprint:
                    raise ValueError("Protocol fingerprint mismatch")
                if row["image"] in images:
                    raise ValueError("Repeated metadata image path")
                images.add(row["image"])
                path = (root / row["image"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Image path escapes collection directory")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("Image checksum mismatch")
                key = (row["person"], row["condition"], row["label"], row["capture_type"])
                hashes[key].append(digest)
                conditions[row["person"]].add(row["condition"])
            except (ValueError, OSError) as exc:
                errors.append({"row": number, "image": row["image"], "error": str(exc)})
    # Count unique bytes per target; identical burst frames do not inflate coverage.
    for (person, condition, label, capture_type), values in hashes.items():
        coverage[(person, condition)][(label, capture_type)] = len(set(values))
    missing = []
    for person, variants in sorted(conditions.items()):
        if len(variants) < 2:
            missing.append({"person": person, "reason": "needs_two_conditions"})
        for condition in sorted(variants):
            for label, capture_type in TARGETS:
                count = coverage[(person, condition)][(label, capture_type)]
                if count < min_per_target:
                    missing.append({"person": person, "condition": condition, "label": label,
                                    "capture_type": capture_type, "unique_images": count})
    return {
        "images": len(images), "people": len(conditions), "person_roles": roles,
        "min_unique_images_per_target": min_per_target,
        "integrity_errors": errors, "missing_coverage": missing,
        "coverage_complete": bool(images) and not errors and not missing,
        "label_review_complete": False, "training_ready": False,
    }


def build_parser(default_protocol):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(default_protocol))
    parser.add_argument("--output", default="data/capture/asl29")
    parser.add_argument("--person", type=component, required=True)
    parser.add_argument("--role", choices=COLLECTION_ROLES, required=True)
    parser.add_argument("--condition", type=component, required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--dry-run", action="store_true")
    return parser
