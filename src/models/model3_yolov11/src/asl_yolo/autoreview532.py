"""User-authorized, retention-oriented automated weak-label review.

Original files, source split, quarantine and holdout are immutable inputs. This
policy does not manufacture human approvals or pretend that MediaPipe verifies ASL.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
from threading import local
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

from .core import digest, file_hash, read_json, write_json
from .dataset29 import load_review
from .preprocess532 import draw_tile, map_padded_points, save_sheet, write_csv, write_jsonl
from .protocol import CLASSES_29, validate_annotation
from .review532 import eligible, event, load_base, materialize_release
from .vision import HandTracker, landmarks_box, xyxy_to_yolo, yolo_to_xyxy

LOCAL = local()


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_policy(path):
    policy = read_json(path)
    auth = policy.get("authorization", {})
    if auth.get("authorized_by") != "user" or auth.get("allow_automated_review") is not True or auth.get("human_review_claimed") is not False:
        raise ValueError("Automated release requires explicit user authorization and honest automated provenance")
    return policy


def valid_box(row, boxes, policy, capture_type=None):
    try:
        kind = capture_type or row["capture_type"]
        validate_annotation(row["label"], kind, boxes)
        if kind == "no_hand":
            return True
        if len(boxes) != 1:
            return False
        _, _, w, h = boxes[0]
        return (min(w, h) >= policy["minimum_dimension"] and
                policy["minimum_area"] <= w * h <= policy["maximum_area"] and
                1 / policy["maximum_aspect_ratio"] <= w / h <= policy["maximum_aspect_ratio"])
    except (ValueError, TypeError, ZeroDivisionError):
        return False


def expanded(box, amount):
    x, y, w, h = box
    a, b = max(0., x - w * (.5 + amount)), max(0., y - h * (.5 + amount))
    c, d = min(1., x + w * (.5 + amount)), min(1., y + h * (.5 + amount))
    return [(a + c) / 2, (b + d) / 2, c - a, d - b]


def overlap(first, second):
    a, b = yolo_to_xyxy(first, 1, 1), yolo_to_xyxy(second, 1, 1)
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.


def decision(row, policy, boxes, state, note, **evidence):
    return dict(state=state, boxes=boxes, sha256=row["sha256"], output_sha256=row["output_sha256"],
                capture_type=row["capture_type"], review_kind="automated", reviewer="ASL29 retention-oriented automated QC",
                method=policy["method"], reviewed_at=utc(), note=note, evidence=evidence)


def rescreen_cached(row, old, policy):
    """Return None only for cases needing fresh extraction; semantic uncertainty stays pending."""
    if row["capture_type"] in {"other_hand", "unresolved"}:
        return decision(row, policy, old["boxes"], "pending", "Requires AI visual confirmation of nothing subtype/pose; not inferred from detector failure.", semantic_review_required=True)
    cross = old.get("evidence", {}).get("crosscheck", {})
    measurement = cross.get("evidence", {})
    counts = measurement.get("hand_counts", [])
    if any(n > 1 for n in counts) or "multiple_hands" in row["flags"]:
        return decision(row, policy, old["boxes"], "pending", "Multiple-hand evidence needs per-object labels or explicit visual correction.", hand_counts=counts)
    if row["capture_type"] == "no_hand":
        if old["state"] == "machine_pass" and old.get("evidence", {}).get("hand_counts") == [0, 0]:
            return decision(row, policy, [], "approved", "Declared background source plus two detector passes; automated review, not a new negative label from a failure.", basis="declared_background_two_passes")
        return decision(row, policy, [], "pending", "Declared background has unresolved hand evidence.")
    if row.get("annotation_state") == "reviewed" and valid_box(row, row["boxes"], policy):
        review = row.get("review_history", [])
        if review and review[-1].get("state") == "approved" and review[-1].get("sha256") == row["sha256"]:
            return decision(row, policy, row["boxes"], "approved", "Preserved previously approved, exact-SHA-bound annotation under authorized automated-review policy.", basis="hash_matched_prior_review", prior_review_kind=review[-1].get("review_kind"))
    if old["state"] == "machine_pass" and valid_box(row, old["boxes"], policy):
        return decision(row, policy, old["boxes"], "approved", "Retained the stricter round's corroborated proposal; no human approval claimed.", basis="strict_round_pass")
    score = measurement.get("iou", measurement.get("best_iou", 0.))
    if math.isfinite(float(score)) and score >= policy["cached_iou_min"] and any(n == 1 for n in counts) and len(old["boxes"]) == 1:
        boxes = [expanded(old["boxes"][0], policy["box_expansion_per_side"])]
        if valid_box(row, boxes, policy):
            return decision(row, policy, boxes, "approved", "Retained under relaxed geometry policy; box may touch image edges. Source folder class retained as a weak label.", basis="relaxed_cached_agreement", iou=score, hand_counts=counts)
    return None


class RetentionTracker(HandTracker):
    def __init__(self, asset, policy):
        import mediapipe as mp
        self.mp, self.video, self.last_timestamp, self.handedness = mp, False, -1, []
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)), num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_hand_detection_confidence=policy["fallback_detection_confidence"],
            min_hand_presence_confidence=policy["fallback_presence_confidence"],
            min_tracking_confidence=.2)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


def inspect_fallback(dataset, row, old, policy):
    import cv2
    path = dataset / row["image"]
    if file_hash(path) != row["output_sha256"]:
        raise ValueError("Image changed before automated review")
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    hits, attempts = [], []
    for mode, border in policy["fallback_border_trials"]:
        frame = rgb[:, ::-1] if mode == "mirror" else rgb
        if border:
            frame = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_REPLICATE)
        hands = LOCAL.tracker.detect(frame)
        attempts.append(dict(mode=mode, border_px=border, hand_count=len(hands)))
        if len(hands) > 1:
            return decision(row, policy, old["boxes"], "pending", "Fresh extractor found multiple hands; no folder class assigned to every hand.", attempts=attempts)
        if not hands:
            continue
        points = map_padded_points(hands[0], row["width"], row["height"], border)
        if mode == "mirror":
            points[:, 0] = 1 - points[:, 0]
        try:
            box = xyxy_to_yolo(landmarks_box(points, row["width"], row["height"], .15), row["width"], row["height"])
            if not valid_box(row, [box], policy):
                continue
        except ValueError:
            continue
        for previous in hits:
            score = overlap(previous, box)
            if score >= policy["fallback_iou_min"]:
                a, b = yolo_to_xyxy(previous, 1, 1), yolo_to_xyxy(box, 1, 1)
                union = xyxy_to_yolo([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], 1, 1)
                boxes = [expanded(union, policy["box_expansion_per_side"])]
                if valid_box(row, boxes, policy):
                    return decision(row, policy, boxes, "approved", "Fresh low-threshold full-frame proposals agree; border coordinates mapped back, source label retained.", basis="fresh_agreement", iou=score, attempts=attempts)
        hits.append(box)
    if len(hits) == 1 and policy["allow_single_pass_proposals"]:
        boxes = [expanded(hits[0], policy["box_expansion_per_side"])]
        if valid_box(row, boxes, policy):
            return decision(row, policy, boxes, "approved", "Single valid fresh proposal retained under permissive policy; lower corroboration, not independently verified ASL ground truth.", basis="single_pass_weak_label", attempts=attempts)
    return decision(row, policy, old["boxes"], "pending", "No reliable box or inconsistent proposals; image retained in master manifest, never relabelled nothing.", attempts=attempts)


def prepare(root, dataset, source_round, output, policy_path, workers):
    policy = load_policy(policy_path)
    manifest, rows = load_base(dataset)
    selected = [r for r in rows if eligible(r)]
    old = load_review(source_round / "evidence.jsonl")["annotations"]
    if set(old) != {r["sample_id"] for r in selected}:
        raise ValueError("Cached round must cover every eligible development image")
    for row in selected:
        d = old[row["sample_id"]]
        if d["sha256"] != row["sha256"] or d["output_sha256"] != row["output_sha256"]:
            raise ValueError("Stale cached evidence")
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(base_fingerprint=manifest["fingerprint"], policy_sha256=file_hash(policy_path),
                     code_sha256=file_hash(Path(__file__)), cached_evidence_sha256=file_hash(source_round / "evidence.jsonl"),
                     asset_sha256=file_hash(root / "assets/hand_landmarker.task"), mediapipe=importlib.metadata.version("mediapipe"))
    meta = output / "auto_round.json"
    if meta.exists() and read_json(meta)["signature"] != signature:
        raise ValueError("Automatic review inputs changed; use a new --output")
    if not meta.exists():
        write_json(meta, {"signature": signature, "policy": policy, "started_at": utc(), "review_kind": "automated"})
        write_json(output / "policy.json", policy)
        (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cache = output / "automatic_decisions.jsonl"
    saved = load_review(cache).get("annotations", {}) if cache.exists() else {}
    by_id = {r["sample_id"]: r for r in selected}
    for key, d in saved.items():
        if key not in by_id or d["sha256"] != by_id[key]["sha256"] or d.get("method") != policy["method"]:
            raise ValueError("Invalid automatic review cache")
    remaining = [r for r in selected if r["sample_id"] not in saved]
    event(output, "automatic_review_started", remaining=len(remaining), eligible=len(selected), authorization=policy["authorization"])
    print(f"Retention review: {len(remaining)}/{len(selected)} images", flush=True)
    trackers = []
    def initialize():
        LOCAL.tracker = None
    def process(row):
        previous = old[row["sample_id"]]
        result = rescreen_cached(row, previous, policy)
        if result is None:
            if LOCAL.tracker is None:
                LOCAL.tracker = RetentionTracker(root / "assets/hand_landmarker.task", policy)
                trackers.append(LOCAL.tracker)
            result = inspect_fallback(dataset, row, previous, policy)
        result["previous_round_state"] = previous["state"]
        result["previous_round_note"] = previous["note"]
        return result
    try:
        with cache.open("a", encoding="utf-8", buffering=1) as stream:
            with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
                for index, (row, d) in enumerate(zip(remaining, pool.map(process, remaining)), 1):
                    stream.write(json.dumps({"sample_id": row["sample_id"], "decision": d}, allow_nan=False) + "\n")
                    saved[row["sample_id"]] = d
                    if index % 500 == 0 or index == len(remaining):
                        print(f"Reviewed {index}/{len(remaining)}", flush=True)
                        event(output, "automatic_review_progress", processed=index, total=len(remaining))
    finally:
        for tracker in trackers:
            tracker.close()
    qc_ids = [i for p in read_json(source_round / "preview_index.json") for i in p["sample_ids"]]
    write_json(output / "qc_sample_ids.json", qc_ids)
    stats = dict(eligible_images=len(selected), decisions=dict(Counter(d["state"] for d in saved.values())),
                 basis=dict(Counter(d["evidence"].get("basis", "pending") for d in saved.values())),
                 qc_gallery_images=len(qc_ids), review_kind="automated", human_review_claimed=False)
    write_json(output / "preliminary_summary.json", stats)
    gallery(dataset, output, rows, saved, qc_ids, "qc")
    # Inspect novel rescues on train only, without using holdout to select parameters.
    groups = {}
    for row in selected:
        d = saved[row["sample_id"]]
        if row["split"] == "train" and d["state"] == "approved" and d["evidence"].get("basis") in {"fresh_agreement", "single_pass_weak_label", "relaxed_cached_agreement"}:
            groups.setdefault((d["evidence"]["basis"], row["label"]), []).append(row["sample_id"])
    rescue_ids = []
    for key, ids in sorted(groups.items()):
        random.Random(digest([42, key])).shuffle(ids)
        rescue_ids.append(ids[0])
    write_json(output / "rescue_sample_ids.json", rescue_ids)
    gallery(dataset, output, rows, saved, rescue_ids, "rescues")
    event(output, "automatic_review_completed", **stats)
    return stats


def gallery(dataset, output, rows, decisions, ids, prefix):
    by_id = {r["sample_id"]: r for r in rows}
    folder = output / "previews"
    folder.mkdir(exist_ok=True)
    pages = []
    for start in range(0, len(ids), 20):
        tiles = []
        for key in ids[start:start + 20]:
            row, d = by_id[key], decisions[key]
            with Image.open(dataset / row["image"]) as image:
                tiles.append(draw_tile(image, d["boxes"], f"{row['label']} {d['state']} {key[:8]}", "lime" if d["state"] == "approved" else "orange"))
        path = folder / f"{prefix}_{start // 20 + 1:03d}.jpg"
        save_sheet(path, tiles)
        pages.append(dict(page=path.relative_to(output).as_posix(), sample_ids=ids[start:start + 20]))
    write_json(output / f"{prefix}_preview_index.json", pages)


def validate_automatic(row, d, policy):
    if not eligible(row) or d.get("sha256") != row["sha256"] or d.get("output_sha256") != row["output_sha256"]:
        raise ValueError("Automatic decision cannot release changed, quarantined or holdout images")
    if d.get("review_kind") != "automated" or not d.get("reviewer") or not d.get("method") or not d.get("reviewed_at") or not d.get("note"):
        raise ValueError("Automatic reviewer, method, note and timestamp must remain explicit")
    if d.get("state") not in {"approved", "pending"}:
        raise ValueError("Unexpected automated state")
    if d["state"] == "approved":
        kind = d.get("capture_type", row["capture_type"])
        if not valid_box(row, d["boxes"], policy, kind):
            raise ValueError("Approved automatic annotation violates geometry or no_hand/other_hand rules")
        if row["capture_type"] in {"other_hand", "unresolved"} and not d.get("semantic_evidence"):
            raise ValueError("Nothing semantics require explicit AI visual evidence")


def finalize(root, dataset, source_round, output, policy_path, release):
    policy = load_policy(policy_path)
    manifest, rows = load_base(dataset)
    by_id = {r["sample_id"]: r for r in rows}
    meta = read_json(output / "auto_round.json")
    if meta["signature"]["base_fingerprint"] != manifest["fingerprint"] or meta["signature"]["policy_sha256"] != file_hash(policy_path):
        raise ValueError("Automatic policy or base changed")
    decisions = load_review(output / "automatic_decisions.jsonl")["annotations"]
    expected = {r["sample_id"] for r in rows if eligible(r)}
    if set(decisions) != expected:
        raise ValueError("Automatic review must cover the complete eligible development population")
    visual_path = output / "ai_visual_decisions.jsonl"
    visual = load_review(visual_path).get("annotations", {}) if visual_path.exists() else {}
    qc_ids = read_json(output / "qc_sample_ids.json")
    if not set(qc_ids) <= set(visual):
        raise ValueError("AI visual review must cover every original QC gallery image")
    if not set(visual) <= expected:
        raise ValueError("Unexpected AI visual decision")
    for key, d in visual.items():
        decisions[key] = d
    selected = {}
    for row in rows:
        key = row["sample_id"]
        if key not in decisions:
            continue
        d = decisions[key]
        validate_automatic(row, d, policy)
        if d["state"] == "approved":
            selected[key] = dict(boxes=d["boxes"], capture_type=d["capture_type"], acceptance=d["evidence"].get("basis", "AI_visual_automated_review"))
    write_jsonl(output / "decisions.jsonl", [{"sample_id": key, "decision": d} for key, d in decisions.items()])
    gallery(dataset, output, rows, decisions, qc_ids, "qc")
    result = materialize_release(dataset, output, release, manifest, rows, selected, {
        "review_policy": policy["method"], "review_kind": "automated", "human_review_claimed": False,
        "annotation_quality": policy["quality_claim"], "authorization": policy["authorization"],
        "automated_decisions_sha256": file_hash(output / "decisions.jsonl"),
        "AI_visual_decisions_sha256": file_hash(visual_path), "policy_sha256": file_hash(policy_path),
    })
    write_json(release / "review_policy.json", policy)
    write_json(release / "source_manifest_reference.json", {"base_fingerprint": manifest["fingerprint"], "master_images": len(rows),
               "base_dataset": str(dataset), "base_manifest": "manifest.jsonl", "all_holdout_and_pending_images_retained": True})
    fields = ["sample_id", "split", "label", "capture_type", "state", "review_kind", "reviewer", "note", "sha256"]
    records = [{**d, "sample_id": key, "split": by_id[key]["split"], "label": by_id[key]["label"]} for key, d in decisions.items()]
    write_csv(output / "review_decisions.csv", records, fields)
    stats = {**result, "total_reviewed": len(decisions), "pending": sum(d["state"] != "approved" for d in decisions.values()),
             "qc_images_kept_in_gallery": len(qc_ids), "qc_approved": sum(decisions[i]["state"] == "approved" for i in qc_ids),
             "qc_pending": sum(decisions[i]["state"] != "approved" for i in qc_ids), "human_review_claimed": False}
    write_json(output / "final_summary.json", stats)
    write_json(release / "review_summary.json", stats)
    report_dir = root / "reports/asl_yolo11n_group_532_auto_v2"
    report_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for page in read_json(output / "qc_preview_index.json"):
        cards.append(f'<section><h2>{page["page"]}</h2><img src="{page["page"]}" alt="AI QC"><p>' + " | ".join(
            f'<a href="{quote(Path(os.path.relpath(dataset / by_id[key]["image"], output)).as_posix())}">{key[:8]}: {decisions[key]["state"]}</a>' for key in page["sample_ids"]) + '</p></section>')
    (output / "index.html").write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>ASL29 QC tự động</title><style>body{margin:30px auto;max-width:1250px;background:#121c26;color:#e8eff4;font:16px/1.6 system-ui}img{max-width:100%}a{color:#7ddafa}section{padding:20px;margin:30px 0;background:#1c2b38}</style><h1>QC tự động — giữ toàn bộ ảnh để truy vết</h1><p>Nhãn tự động có kiểm tra trực quan bằng AI theo yêu cầu người dùng. Không phải duyệt thủ công hoặc kiểm định chuyên gia ASL.</p>' + "\n".join(cards) + '</html>', encoding="utf-8")
    text = f"""# Duyệt tự động ASL29 — tiêu chí ưu tiên giữ ảnh

Người dùng đã yêu cầu tự động duyệt, bớt khắt khe và giữ ảnh QC tối đa. Policy riêng: `{policy_path.relative_to(root).as_posix()}`. Bản gốc và vòng QC01 được giữ nguyên.

- Đã xét {len(decisions):,} ảnh development không bị cách ly. Train/val phát hành: **{result['counts']['train']:,}/{result['counts']['val']:,}**. Còn pending: **{stats['pending']:,}**.
- Giữ **toàn bộ {len(qc_ids)} ảnh** trong gallery QC; approved **{stats['qc_approved']}**, pending **{stats['qc_pending']}**. Không xóa ảnh khó hoặc loại cả nhóm vì một mẫu lỗi.
- `review_kind=automated`, `human_review_claimed=false`. Nhãn nguồn được giữ; đây là weak labels có kiểm tra AI theo mẫu, không phải kiểm định chuyên gia ngôn ngữ ký hiệu.
- IoU cache/fallback tối thiểu 0.50, chiều tối thiểu 3.5%, diện tích 0.001..0.98, nới box 10%, cho chạm biên. Bỏ yêu cầu margin 0.008 và chiều 12%/IoU 0.88 cũ. Không chỉnh cấu hình dựa trên test.
- Tái dùng annotation đã duyệt khi đúng hash; phần thiếu/thất bại được thử MediaPipe ở ngưỡng 0.2, ảnh gốc/lật/border replicate 50/80px, ánh xạ về ảnh gốc. Single-pass được ghi rõ mức corroboration thấp. Không crop hay đổi nhãn thành nothing vì detector thất bại.
- `nothing/no_hand` chỉ dùng nhãn rỗng; positive class 27 phải là other_hand có box và bằng chứng trực quan. Tư thế gần H/B hoặc nhiều tay không chắc vẫn pending.
- Giữ nguyên split nguồn đề xuất 5/3/2, quarantine và holdout Bao. Run subset mới được khóa bằng fingerprint; không tuyên bố trưởng nhóm đã phê duyệt split. Session chưa biết vẫn unknown.
- `training_ready={str(result['training_ready']).lower()}` theo policy tự động đã được người dùng cho phép, đủ positive train boxes 29 lớp và loader/hash/leakage checks đạt. `evaluation_ready=false`: holdout thiếu ground truth, validation vẫn có thể thiếu positive nothing từ session độc lập. Không chạy train hoặc đánh giá model trong tác vụ này.

Dataset: `{release.relative_to(root).as_posix()}`. Dùng `draft/data.yaml` cho bước train sau; `training_plan.json` vẫn tắt allow_training/final_test.
Fingerprint: `{result['fingerprint']}`. Danh sách quyết định, hash, lý do và thời điểm: `decisions.jsonl`, `review_decisions.csv`; gallery: `index.html`.

```powershell
.\\.venv\\Scripts\\python.exe scripts/auto_review_group_532.py --prepare --workers 4
# ai_visual_decisions.jsonl là các quyết định AI gắn hash từ việc xem contact sheet/toàn ảnh; giữ file này để tái lập.
.\\.venv\\Scripts\\python.exe scripts/auto_review_group_532.py --finalize
```

Không ghi đè release đã hoàn tất. Tái tạo vào --release mới. Ảnh và nhãn phát hành có checksum; toàn bộ master, pending và holdout tham chiếu bản gốc vẫn giữ đủ. Không có accuracy/F1/mAP được tạo.
"""
    (report_dir / "AUTO_REVIEW_REPORT.md").write_text(text, encoding="utf-8")
    (release / "README.md").write_text(text, encoding="utf-8")
    event(output, "automated_release_completed", **stats)
    return stats
