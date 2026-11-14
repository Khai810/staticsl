"""Single-hand association and an honest per-letter detector score display.

These webcam settings do not change the saved validation/test inference policy.
"""
from __future__ import annotations

import numpy as np


def box_iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    box = np.asarray(box, dtype=float)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


class SingleHandLock:
    """Keep the first observed hand; never automatically acquire after a lost lock.

    Match geometry plus confident handedness, not MediaPipe's changing list order.
    Ambiguous crossings are rejected. This is spatial tracking, not biometric ID.
    """
    def __init__(self, max_gap=.8):
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.target = None
        self.last_seen = None
        self.lost = False

    def update(self, candidates, now):
        if self.lost:
            return None
        if self.target is None:
            if not candidates:
                return None
            self.target = dict(candidates[0])
            self.last_seen = now
            return self.target
        if now - self.last_seen > self.max_gap:
            self.lost = True
            return None
        previous = np.asarray(self.target["box"], dtype=float)
        center = (previous[:2] + previous[2:]) / 2
        size = np.maximum(previous[2:] - previous[:2], .01)
        matches = []
        for candidate in candidates:
            box = np.asarray(candidate["box"], dtype=float)
            distance = float(np.linalg.norm((box[:2] + box[2:]) / 2 - center))
            ratio = (np.maximum(box[2:] - box[:2], .001) / size).max()
            inverse_ratio = (size / np.maximum(box[2:] - box[:2], .001)).max()
            label, certainty = candidate.get("handedness", (None, 0.))
            old_label, old_certainty = self.target.get("handedness", (None, 0.))
            if label and old_label and label != old_label and min(certainty, old_certainty) >= .8:
                continue
            if distance > min(.25, max(.10, float(np.linalg.norm(size)) * .65)) or max(ratio, inverse_ratio) > 2.5:
                continue
            matches.append((distance, candidate))
        matches.sort(key=lambda item: item[0])
        if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < .035):
            return None
        candidate = dict(matches[0][1])
        if "handedness" not in candidate and "handedness" in self.target:
            candidate["handedness"] = self.target["handedness"]
        self.target = candidate
        self.last_seen = now
        return self.target


def score_predictor_class():
    # Lazy import: tracking and rendering tests do not need torch/Ultralytics.
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops

    class ScorePredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            nc = len(self.model.names)
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != nc + 4:
                raise RuntimeError("Per-letter panel requires YOLO11 detection output [1, 4+classes, anchors]")
            # Preserve independent sigmoid scores BEFORE NMS discards other labels.
            self.letter_scores = raw[0, 4:4 + nc].transpose(0, 1).detach().cpu().numpy().copy()
            boxes = ops.xywh2xyxy(raw[0, :4].transpose(0, 1)).clone()
            self.letter_boxes = ops.scale_boxes(img.shape[2:], boxes, orig_imgs[0].shape).detach().cpu().numpy()
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    return ScorePredictor


def select_hand_scores(boxes, scores, target_box, other_boxes=()):
    """Return all class scores at ONE anchor associated with the locked hand.

    No softmax/renormalization and no maximum over unrelated hands/background.
    A below-threshold anchor remains a diagnostic candidate, not a detection.
    """
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    if len(boxes) == 0 or target_box is None:
        return np.zeros(scores.shape[1], dtype=float), None
    target = np.asarray(target_box, dtype=float)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    overlaps = box_iou(target, boxes)
    valid = ((centers >= target[:2]).all(1) & (centers <= target[2:]).all(1) & (overlaps >= .15))
    for other in other_boxes:
        # Exclude a competing hand's anchors, including overlap during a crossing.
        valid &= overlaps > box_iou(other, boxes)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return np.zeros(scores.shape[1], dtype=float), None
    index = indices[np.argmax(scores[indices].max(1))]
    return scores[index].copy(), boxes[index].copy()


def hand_crop(frame, box):
    """Optional unannotated BGR crop; no landmark pixels/vectors enter YOLO."""
    h, w = frame.shape[:2]
    box = np.asarray(box, dtype=float) * [w, h, w, h]
    center = (box[:2] + box[2:]) / 2
    size = max(*(box[2:] - box[:2])) * 1.35
    low = np.maximum(0, np.floor(center - size / 2)).astype(int)
    high = np.minimum([w, h], np.ceil(center + size / 2)).astype(int)
    return frame[low[1]:high[1], low[0]:high[0]], low


def draw_dashboard(frame, classes, scores, target, predicted_box, threshold, fps, state, input_mode):
    import cv2
    from .demo import CONNECTIONS

    h, w = frame.shape[:2]
    height = max(h, 480) + 112
    canvas = np.full((height, w + 420, 3), (25, 23, 21), dtype=np.uint8)
    canvas[62:62 + h, :w] = frame
    white, muted, green, amber = (235, 239, 245), (155, 163, 176), (123, 226, 101), (65, 192, 255)

    def text(value, x, y, scale=.5, color=white, thickness=1):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    text("ASL  |  SINGLE HAND", 18, 28, .65, white, 2)
    text(f"{state}   |   {input_mode.upper()}   |   {fps:.1f} FPS", 18, 49, .43, muted)
    best = int(np.argmax(scores)) if scores is not None else None
    confidence = float(scores[best]) if best is not None else 0.
    accepted = target is not None and confidence >= threshold and predicted_box is not None
    if target is not None:
        box = np.asarray(target["box"]) * [w, h, w, h]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1 + 62), (x2, y2 + 62), green, 2)
        text("HAND 1 - LOCKED", x1, max(78, y1 + 54), .43, green)
        for hand in [target.get("landmarks")]:
            if hand is None:
                continue
            points = [(int(p[0] * w), int(p[1] * h) + 62) for p in hand]
            for a, b in CONNECTIONS:
                cv2.line(canvas, points[a], points[b], (220, 187, 64), 1, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 2, (235, 226, 99), -1)
        if accepted:
            text(f"{classes[best]}  {confidence:.1%}", 18, 103, 1., green, 2)
        elif best is not None and confidence > 0:
            text(f"Candidate {classes[best]}: {confidence:.1%} (LOW)", 18, 100, .65, amber, 2)
        else:
            text("NO_DETECTION", 18, 100, .7, amber, 2)
    else:
        text("LOCK LOST - press R" if state == "LOCK LOST" else "Show one hand to begin", 18, 100, .65, amber, 2)
    text("Q / Esc: quit    R: reset hand    C: full / hand crop", 14, height - 30, .43, muted)
    text(f"[ / ]: confidence threshold   {threshold:.0%}   |   24 static letters (no J/Z)", 14, height - 10, .43, muted)

    x0 = w + 18
    text("CONFIDENCE BY LETTER", x0, 30, .59, white, 2)
    text("One candidate box on the locked hand", x0, 53, .43, muted)
    for i, label in enumerate(classes):
        col, row = divmod(i, 12)
        x, y = x0 + col * 198, 83 + row * 35
        value = float(scores[i]) if scores is not None else 0.
        color = green if i == best and accepted else amber if i == best and value > 0 else muted
        text(label, x, y, .56, color, 2)
        percent = f"{value * 100:5.2f}%" if value < .01 else f"{value * 100:5.1f}%"
        text(percent if scores is not None else "  --", x + 101, y, .47, color)
        cv2.rectangle(canvas, (x + 25, y - 10), (x + 94, y - 1), (60, 57, 54), -1)
        if value > 0:
            cv2.rectangle(canvas, (x + 25, y - 10), (x + 25 + round(69 * value), y - 1), color, -1)
    text("Raw YOLO scores; not a 100% distribution.", x0, height - 50, .42, muted)
    text("Low score = uncertain; no forced letter.", x0, height - 30, .42, muted)
    text("R releases the current hand lock.", x0, height - 10, .42, muted)
    return canvas
