"""Label and annotation contract for the new 29-class detection experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import string

CLASSES_29 = tuple(string.ascii_uppercase) + ("del", "nothing", "space")
COLLECTION_ROLES = ("development", "external_test")


def load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "asl_detection_protocol":
        raise ValueError("Expected the ASL detection protocol, not a prototype training config")
    if protocol.get("task") != "detect" or tuple(protocol.get("classes", [])) != CLASSES_29:
        raise ValueError("Detector classes must match the group's 29-label order exactly")
    if tuple(protocol.get("collection_roles", [])) != COLLECTION_ROLES:
        raise ValueError("Collection roles must be development and external_test")
    policy = protocol.get("nothing_policy", {})
    if policy.get("no_hand", {}).get("requires_box") is not False:
        raise ValueError("No-hand images are background images without object boxes")
    if policy.get("other_hand", {}).get("requires_box") is not True:
        raise ValueError("The learned nothing object class requires reviewed hand boxes")
    if policy.get("model_mistakes_are_ground_truth_nothing") is not False:
        raise ValueError("Ground-truth labels cannot be assigned from model failures")
    return protocol


def protocol_fingerprint(protocol):
    content = json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def annotation_spec(label, capture_type="known_hand"):
    if label not in CLASSES_29:
        raise ValueError(f"Unknown ASL label: {label}")
    if label == "nothing":
        if capture_type not in ("no_hand", "other_hand"):
            raise ValueError("nothing requires capture_type=no_hand or other_hand")
        return {
            "label": label,
            "class_id": CLASSES_29.index(label),
            "capture_type": capture_type,
            "requires_box": capture_type == "other_hand",
            "annotation_policy": "background_empty_label" if capture_type == "no_hand" else "reviewed_nothing_hand_box",
        }
    if capture_type != "known_hand":
        raise ValueError("A named ASL sign must use capture_type=known_hand")
    return {
        "label": label,
        "class_id": CLASSES_29.index(label),
        "capture_type": capture_type,
        "requires_box": True,
        "annotation_policy": "reviewed_known_hand_box",
    }


def validate_annotation(label, capture_type, boxes):
    """Validate reviewed, normalized xywh annotations, including empty backgrounds."""
    spec = annotation_spec(label, capture_type)
    if bool(boxes) != spec["requires_box"]:
        raise ValueError("Object/background annotation does not match the capture type")
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("A box requires four finite normalized xywh coordinates")
        x, y, width, height = map(float, box)
        if not (0 < width <= 1 and 0 < height <= 1):
            raise ValueError("Box width and height must be positive and at most one")
        # Normalization and serialization can move a clipped edge by roundoff.
        tolerance = 1e-9
        if not (-tolerance <= x - width / 2 <= x + width / 2 <= 1 + tolerance and
                -tolerance <= y - height / 2 <= y + height / 2 <= 1 + tolerance):
            raise ValueError("The entire box must be inside the image")
    return spec


def resolve_image_prediction(detections, threshold):
    """Map label/confidence detections to one output while preserving rejection cause."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    accepted = []
    for detection in detections:
        label = detection["label"]
        confidence = float(detection["confidence"])
        if label not in CLASSES_29 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Detection label/confidence is invalid")
        if confidence >= threshold:
            accepted.append((confidence, label))
    if not accepted:
        return {"label": "nothing", "confidence": None, "reason": "no_accepted_detection"}
    confidence, label = max(accepted, key=lambda item: item[0])
    return {
        "label": label,
        "confidence": confidence,
        "reason": "explicit_nothing_detection" if label == "nothing" else "recognized_sign",
    }
