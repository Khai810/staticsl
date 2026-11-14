"""Prepare and audit a draft ASL29 detection dataset from the pinned group split."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from functools import partial
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import imagehash
import numpy as np
from PIL import Image, ImageOps
import yaml

from .capture29 import TARGETS, claim_person_role
from .core import digest, file_hash, read_json, write_json
from .data import duplicate_groups
from .protocol import CLASSES_29, annotation_spec, protocol_fingerprint, validate_annotation
from .vision import HandTracker, landmarks_box, xyxy_to_yolo


ARCHIVE_TRAIN = "asl_alphabet_train/asl_alphabet_train/"
GROUP_TRAIN = "data/raw/asl-alphabet/" + ARCHIVE_TRAIN
SPLITS = ("train", "val", "group_test")
FIT_SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PADDING = .15
NEAR_DISTANCE = 2
SPLIT_PRIORITY = {"external_test": 0, "group_test": 1, "val": 2, "train": 3}


def git_blob(content):
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def pinned_tree(repository, commit):
    result = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", "-r", "-z", commit, "--",
         "data/raw/asl-alphabet", "data/hybrid_manifest"],
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    tree = {}
    for entry in result.stdout.rstrip("\0").split("\0"):
        info, path = entry.split("\t", 1)
        mode, kind, object_id = info.split()
        if kind == "blob":
            tree[path] = object_id
    return tree


def sample(source, source_path, split, label, capture_type, **metadata):
    return {
        "sample_id": digest([source, source_path])[:24],
        "source": source, "source_path": source_path, "split": split,
        **annotation_spec(label, capture_type), **metadata,
        "image_status": "unverified", "review_status": "pending",
        "boxes": [], "flags": [], "exclusions": [],
    }


def image_details(content):
    with Image.open(BytesIO(content)) as image:
        image.load()
        exif = image.getexif()
        exif_ifd = exif.get_ifd(34665)
        details = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "format": image.format,
            "exif_orientation": exif.get(274),
            "exif_datetime_original": exif_ifd.get(36867) or exif.get(36867),
            "exif_subsec_time_original": exif_ifd.get(37521) or exif.get(37521),
            "exif_offset_time_original": exif_ifd.get(36881) or exif.get(36881),
            "exif_software": exif.get(305),
        }
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    if min(width, height) < 10 or details["format"] not in {"JPEG", "PNG"}:
        raise ValueError("Unsupported image format or dimensions for YOLO")
    details.update(width=width, height=height,
                   pixel_sha256=hashlib.sha256(f"{width}x{height}\0".encode() + rgb.tobytes()).hexdigest(),
                   phash=str(imagehash.phash(rgb)))
    return details, rgb


def propose_box(row, points):
    points = np.asarray(points, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite raw xyz landmarks")
    xy = points[:, :2]
    if (xy < 0).any() or (xy > 1).any():
        row["flags"].append("landmark_outside_image")
    width, height = row["width"], row["height"]
    xyxy = landmarks_box(points, width, height, PADDING)
    box = xyxy_to_yolo(xyxy, width, height)
    validate_annotation(row["label"], row["capture_type"], [box])
    if min(xyxy[0], xyxy[1], width - xyxy[2], height - xyxy[3]) < 1:
        row["flags"].append("box_touches_image_edge")
    if box[2] * box[3] < .01 or box[2] * box[3] > .85:
        row["flags"].append("unusual_box_area")
    row["boxes"].append(box)


def materialize(row, content, output, rgb):
    split = row["split"]
    scope = "draft" if split in FIT_SPLITS else "holdout"
    extension = ".jpg" if row["format"] == "JPEG" else ".png"
    relative = Path(scope, "images", split, row["sample_id"] + extension)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if row["exif_orientation"] not in (1, None):
        # Boxes and hashes of decoded pixels use the displayed orientation.
        relative = relative.with_suffix(".png")
        path = output / relative
        rgb.save(path)
        row["output_sha256"] = file_hash(path)
    else:
        path.write_bytes(content)
        row["output_sha256"] = row["sha256"]
    row.update(image_status="readable", image=relative.as_posix())


def group_rows(archive_path, csv_root, tree, output):
    rows, seen, vectors = [], set(), defaultdict(list)
    inventory = {"csv_git_blobs": {}, "archive_members": {}, "raw_test_aliases": [],
                 "landmark_duplicates_cross_split": []}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(set(members)) != len(members):
            raise ValueError("Archive contains duplicate member paths")
        members = set(members)
        if not any(name.startswith(ARCHIVE_TRAIN) for name in members):
            raise ValueError("Expected the grassknoted/asl-alphabet archive, not the prototype ZIP")
        inventory["archive_members"] = dict(Counter(name.split("/")[0] for name in members
                                                    if not name.endswith("/")))

        def add(row, entry, raw_landmarks=None):
            rows.append(row)
            expected = tree.get(row["source_path"])
            row["git_blob_sha1"] = expected
            if expected is None:
                row.update(image_status="missing_from_pinned_tree")
                return
            try:
                content = archive.read(entry)
                if git_blob(content) != expected:
                    raise ValueError("Image bytes differ from pinned Git blob")
                details, rgb = image_details(content)
                row.update(details)
                materialize(row, content, output, rgb)
            except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
                row.update(image_status="source_error", error=str(exc))
                return
            if raw_landmarks is not None:
                row["box_source"] = "group_raw_landmarks"
                try:
                    if row["exif_orientation"] not in (1, None):
                        raise ValueError("Verify landmark orientation manually")
                    propose_box(row, raw_landmarks)
                except ValueError as exc:
                    row["flags"].append("invalid_landmarks_or_box")
                    row["annotation_error"] = str(exc)
            else:
                row["box_source"] = "raw_nothing_background_candidate"

        for split in SPLITS:
            csv_name = "test" if split == "group_test" else split
            csv_file = csv_root / f"{csv_name}.csv"
            expected = tree[f"data/hybrid_manifest/{csv_file.name}"]
            if git_blob(csv_file.read_bytes()) != expected:
                raise ValueError(f"{csv_file.name} differs from pinned Git blob")
            inventory["csv_git_blobs"][split] = expected
            with csv_file.open(encoding="utf-8-sig", newline="") as stream:
                for record in csv.DictReader(stream):
                    label, filename = record["label"], record["source_file"]
                    if label == "nothing" or Path(filename).name != filename or "/" in filename or "\\" in filename:
                        raise ValueError(f"Invalid pinned manifest mapping: {label}/{filename}")
                    key = f"{label}/{filename}"
                    if (split, key) in seen:
                        raise ValueError(f"Repeated source key in {split}: {key}")
                    seen.add((split, key))
                    row = sample("group_csv", GROUP_TRAIN + key, split, label, "known_hand")
                    # A repeated source key across splits must still have a distinct audit row.
                    row["sample_id"] = digest([row["source"], split, row["source_path"]])[:24]
                    try:
                        raw = np.array([float(record[f"r{i}"]) for i in range(63)]).reshape(21, 3)
                    except (ValueError, TypeError):
                        raw = np.empty((0, 3))
                    if raw.shape == (21, 3) and np.isfinite(raw).all():
                        vectors[hashlib.sha256(raw.tobytes()).digest()].append(row)
                    add(row, ARCHIVE_TRAIN + key, raw)

        for path in sorted(tree):
            if path.startswith(GROUP_TRAIN + "nothing/"):
                row = sample("group_raw_background", path, "train", "nothing", "no_hand")
                add(row, path.removeprefix("data/raw/asl-alphabet/"))

        raw_by_blob = {blob: path for path, blob in tree.items() if path.startswith(GROUP_TRAIN)}
        for path, blob in sorted(tree.items()):
            if path.startswith("data/raw/asl-alphabet/asl_alphabet_test/") and blob in raw_by_blob:
                inventory["raw_test_aliases"].append([path, raw_by_blob[blob]])
        for group in vectors.values():
            if len({row["split"] for row in group}) > 1:
                inventory["landmark_duplicates_cross_split"].append([
                    {key: row[key] for key in ("split", "source_path", "image_status")} for row in group
                ])
        mapped = [row for row in rows if row["source"] == "group_csv" and row.get("git_blob_sha1")]
        number = {}
        for row in mapped:
            match = re.fullmatch(re.escape(row["label"]) + r"(\d+)\.jpg", Path(row["source_path"]).name)
            if match:
                number[(row["label"], int(match[1]))] = row["split"]
        inventory["adjacent_filename_pairs_cross_split"] = dict(Counter(
            "/".join(sorted((split, next_split)))
            for (label, index), split in number.items()
            if (next_split := number.get((label, index + 1))) and next_split != split
        ))
    return rows, inventory


def capture_rows(inbox, asset, output, proposals=None, freeze_base_fit=False):
    rows, roles = [], {}
    proposals = proposals or {}
    paths = sorted(path for path in inbox.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        return rows
    with ExitStack() as stack:
        tracker = None
        for path in paths:
            relative = path.relative_to(inbox)
            role, person, condition, *target, _filename = relative.parts
            roles = claim_person_role(roles, person, role)
            if person.casefold() == "bao" and role != "external_test":
                raise ValueError("Bao belongs to external_test")
            label, capture_type = tuple(target) if len(target) == 2 else (target[0], "known_hand")
            if (label, capture_type) not in TARGETS or len(target) not in (1, 2):
                raise ValueError(f"Unexpected capture path: {relative}")
            split = "external_test" if role == "external_test" else "train"
            row = sample("capture_inbox", relative.as_posix(), split, label, capture_type,
                         role=role, person=person, condition=condition,
                         collection_group=f"{role}/{person}/{condition}")
            rows.append(row)
            try:
                content = path.read_bytes()
                details, rgb = image_details(content)
                row.update(details)
                materialize(row, content, output, rgb)
            except (OSError, ValueError) as exc:
                row.update(image_status="source_error", error=str(exc))
                continue
            if not row["requires_box"]:
                row["box_source"] = "capture_background_candidate"
                continue
            previous = proposals.get(row["sample_id"])
            if previous and previous.get("sha256") == row["sha256"]:
                boxes = previous["boxes"]
                if boxes:
                    validate_annotation(label, capture_type, boxes)
                row.update(boxes=boxes, flags=list(previous["flags"]), box_source=previous["box_source"])
                if "pose_check" in previous:
                    row["pose_check"] = previous["pose_check"]
                if row.get("pose_check") == "needs_pose_review_similar_to_group_H":
                    if "needs_pose_review" not in row["flags"]:
                        row["flags"].append("needs_pose_review")
                    row["exclusions"].append("capture_pose_pending_review")
                if freeze_base_fit and split in FIT_SPLITS:
                    row["exclusions"].extend(reason for reason in previous.get("exclusions", [])
                                             if reason.startswith(("near_", "exact_")))
                    if "near_match" in previous:
                        row["near_match"] = previous["near_match"]
                continue
            row["box_source"] = "mediapipe_landmarks"
            try:
                if tracker is None:
                    tracker = stack.enter_context(HandTracker(asset))
                hands = tracker.detect(np.asarray(rgb))
                if len(hands) != 1:
                    row["flags"].append("mediapipe_no_hand" if not hands else "mediapipe_multiple_hands")
                for points in hands:
                    propose_box(row, points)
            except (ValueError, RuntimeError) as exc:
                row["flags"].append("mediapipe_box_error")
                row["annotation_error"] = str(exc)
    return rows


def reuse_group_snapshot(base, protocol, output, freeze_base_fit=False, *, include_background=True):
    manifest = read_json(base / "manifest.json")
    if manifest["protocol_sha256"] != protocol_fingerprint(protocol):
        raise ValueError("Base dataset uses a different protocol")
    rows, capture_proposals = [], {}
    transfers = []
    for row in manifest["rows"]:
        if row["source"] == "capture_inbox":
            if row["image_status"] == "readable":
                capture_proposals[row["sample_id"]] = row
            continue
        if row["source"] not in {"group_csv", "group_raw_background"}:
            raise ValueError(f"Unexpected base source: {row['source']}")
        if not include_background and row["source"] == "group_raw_background":
            continue
        if row["image_status"] == "readable":
            relative = Path(row["image"])
            expected = ("draft" if row["split"] in FIT_SPLITS else "holdout", "images", row["split"])
            if relative.parts[:3] != expected or len(relative.parts) != 4 or relative.stem != row["sample_id"]:
                raise ValueError(f"Unexpected base image path: {relative}")
            target = output / relative
            transfers.append((base / relative, target))
        if not freeze_base_fit:
            row["exclusions"] = []
            row.pop("near_match", None)
        row.pop("yolo_label", None)
        row.pop("near_duplicate_group", None)
        if row["review_status"] == "rejected" and "annotation_rejected" not in row["exclusions"]:
            row["exclusions"].append("annotation_rejected")
        rows.append(row)

    def stage_image(paths):
        source, target = paths
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)

    for directory in {target.parent for _, target in transfers}:
        directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(stage_image, transfers):
            pass
    audit = read_json(base / "source_audit.json")
    inventory = {key: value for key, value in audit.items() if key not in {"duplicates", "source_errors"}}
    return rows, inventory, capture_proposals


def apply_reviews(rows, review, allow_automated=False):
    decisions = dict(review.get("annotations", {}))
    for row in rows:
        decision = decisions.pop(row["sample_id"], None)
        if decision is None:
            continue
        if (row["image_status"] != "readable" or decision.get("sha256") != row["sha256"]
                or not decision.get("reviewer", "").strip()):
            raise ValueError(f"Invalid or stale annotation decision: {row['sample_id']}")
        kind = decision.get("review_kind", "human")
        if kind not in {"human", "automated"}:
            raise ValueError(f"Unexpected review kind: {kind}")
        if kind == "automated" and (not allow_automated or not decision.get("method", "").strip()):
            raise ValueError("Automated reviews need explicit opt-in and a documented method")
        state = decision["state"]
        if state == "rejected":
            if not decision.get("note", "").strip():
                raise ValueError("Rejected annotations require a reason")
            row["exclusions"].append("annotation_rejected")
        elif state == "approved":
            if (row["flags"] or row["capture_type"] != "known_hand") and not decision.get("note", "").strip():
                raise ValueError("Explain how flags and the special capture type were confirmed")
            boxes = decision["boxes"]
            validate_annotation(row["label"], row["capture_type"], boxes)
            row.update(boxes=boxes, box_source=f"{kind}_review")
            row["exclusions"] = [reason for reason in row["exclusions"]
                                 if reason not in {"capture_pose_pending_review", "annotation_rejected"}]
        else:
            raise ValueError(f"Unexpected review state: {state}")
        row.update(review_status=state, review=decision)
    if decisions:
        raise ValueError(f"Unknown reviewed sample IDs: {list(decisions)[:5]}")


def load_review(path):
    if path is None:
        return {}
    if path.suffix.lower() != ".jsonl":
        return read_json(path)
    annotations = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entry = json.loads(line)
                annotations[entry["sample_id"]] = entry["decision"]
    return {"annotations": annotations}


def split_key(row):
    return row.get("collection_group", row["split"]) if row["split"] == "external_test" else row["split"]


def quarantine_near_matches(rows):
    """Retain higher-priority splits and log one direct pHash witness per exclusion."""
    index = defaultdict(list)
    # At distance <= 2, at least one of three disjoint chunks matches exactly.
    chunks = ((0, 22), (22, 21), (43, 21))

    def keys(value):
        return [(offset, (value >> offset) & ((1 << width) - 1)) for offset, width in chunks]

    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    quarantine = []
    for split in SPLIT_PRIORITY:
        retained = []
        for row in by_split[split]:
            value = int(row["phash"], 16)
            parts = keys(value)
            match = None
            if split in FIT_SPLITS:
                for key in parts:
                    for candidate_hash, candidate in index.get(key, []):
                        distance = (value ^ candidate_hash).bit_count()
                        if distance <= NEAR_DISTANCE:
                            match = {"sample_id": candidate["sample_id"], "split": candidate["split"],
                                     "distance": distance}
                            break
                    if match:
                        break
            if match:
                row["exclusions"].append("near_cross_split_quarantine")
                row["near_match"] = match
                quarantine.append({"sample_id": row["sample_id"], "split": split, "witness": match})
            else:
                retained.append((value, parts, row))
        for value, parts, row in retained:
            for key in parts:
                index[key].append((value, row))
    return quarantine


def duplicate_audit(rows, review, quarantine_near=False):
    readable = [row for row in rows if row["image_status"] == "readable"]
    result = {}
    for field in ("sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in readable:
            groups[row[field]].append(row)
        repeats = [group for group in groups.values() if len(group) > 1]
        cross_split = [group for group in repeats if len({split_key(row) for row in group}) > 1]
        result[field] = {
            "duplicate_groups": len(repeats),
            "cross_split_groups": [[row["sample_id"] for row in group] for group in cross_split],
        }
        for group in repeats:
            winner = min(SPLIT_PRIORITY[row["split"]] for row in group)
            conflict = len({(row["label"], row["capture_type"]) for row in group}) > 1
            for row in group:
                if row["split"] in FIT_SPLITS and (conflict or SPLIT_PRIORITY[row["split"]] > winner):
                    row["exclusions"].append("exact_conflict" if conflict else "exact_cross_split_duplicate")

    active = [row for row in readable if not row["exclusions"]]
    if quarantine_near:
        frozen = [
            {"sample_id": row["sample_id"], "split": row["split"], "witness": row["near_match"],
             "frozen_from_base": True}
            for row in readable if "near_cross_split_quarantine" in row["exclusions"]
        ]
        result["near_quarantine"] = frozen + quarantine_near_matches(active)
        active = [row for row in active if not row["exclusions"]]
    # Reuse the prototype's perceptual hash grouping, without its split assignment.
    candidates = [{"sample_id": row["sample_id"], "class_name": row["label"],
                   "sha256": row["sha256"], "phash": row["phash"]} for row in active]
    near_groups = defaultdict(list)
    for row, candidate in zip(active, duplicate_groups(candidates, NEAR_DISTANCE)):
        row["near_duplicate_group"] = candidate["duplicate_group"]
        near_groups[candidate["duplicate_group"]].append(row)
    cross = []
    decisions = review.get("near_duplicates", {})
    for group in near_groups.values():
        partitions = {split_key(row) for row in group}
        if len(partitions) < 2:
            continue
        key = digest(sorted((row["sample_id"], row["sha256"], split_key(row)) for row in group))
        decision = decisions.get(key)
        if decision and (decision.get("state") != "distinct" or not decision.get("reviewer", "").strip()
                         or not decision.get("note", "").strip()):
            raise ValueError(f"Invalid near-duplicate decision: {key}")
        cross.append({
            "fingerprint": key, "partitions": sorted(partitions),
            "sample_ids": [row["sample_id"] for row in group],
            "touches_development": any(row["split"] in FIT_SPLITS for row in group),
            "review": decision,
        })
    result["perceptual_hash"] = {"distance": NEAR_DISTANCE, "cross_split_groups": cross}
    return result


def write_yolo_labels(staging, rows):
    lists = {split: [] for split in FIT_SPLITS}
    for row in rows:
        if row["image_status"] != "readable" or row["exclusions"]:
            continue
        if row["requires_box"] and not row["boxes"]:
            continue
        validate_annotation(row["label"], row["capture_type"], row["boxes"])
        image = Path(row["image"])
        label = Path(image.parts[0], "labels", *image.parts[2:]).with_suffix(".txt")
        label_path = staging / label
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(
            f"{row['class_id']} " + " ".join(f"{v:.10f}" for v in box) + "\n" for box in row["boxes"]
        ), encoding="ascii")
        row["yolo_label"] = label.as_posix()
        if row["split"] in FIT_SPLITS:
            lists[row["split"]].append("./" + image.relative_to("draft").as_posix())
    return lists


def write_fit_config(staging, destination, classes, lists, comment=""):
    draft = staging / "draft"
    draft.mkdir(exist_ok=True)
    for split, paths in lists.items():
        (draft / f"{split}.txt").write_text("".join(path + "\n" for path in paths), encoding="utf-8")
    config = {
        "path": (destination / "draft").as_posix(), "train": "train.txt", "val": "val.txt",
        "names": dict(enumerate(classes)),
    }
    (draft / "data.yaml").write_text(comment + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_dataset(staging, destination, protocol, rows, inventory, duplicates, inputs):
    lists = write_yolo_labels(staging, rows)
    validation = summarize_dataset(rows, duplicates)
    comment = "# Draft with proposals. Consult validation.json before use.\n" if validation["blockers"] else ""
    write_fit_config(staging, destination, protocol["classes"], lists, comment)
    write_json(staging / "manifest.json", {
        "schema_version": 1, "protocol_sha256": protocol_fingerprint(protocol), "classes": protocol["classes"],
        "source": protocol["source"], "inputs": inputs, "rows": rows,
    })
    write_json(staging / "source_audit.json", {
        **inventory, "duplicates": duplicates, "source_errors": [
            {k: row.get(k) for k in ("split", "source_path", "image_status", "error")}
            for row in rows if row["image_status"] != "readable"
        ],
    })
    return validation


def summarize_dataset(rows, duplicates):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    counts = {}
    for split, samples in by_split.items():
        readable = [row for row in samples if row["image_status"] == "readable"]
        counts[split] = {
            "rows": len(samples), "readable": len(readable),
            "readable_by_class": {label: sum(row["label"] == label for row in readable)
                                  for label in CLASSES_29},
            "with_yolo_label": sum("yolo_label" in row for row in samples),
            "positive_boxes_by_class": dict(Counter(row["label"] for row in samples
                                                   if row.get("yolo_label") and row["boxes"])),
            "reviewed_positives_by_class": dict(Counter(row["label"] for row in samples
                                                       if row.get("yolo_label") and row["boxes"]
                                                       and row["review_status"] == "approved")),
            "nothing_subtypes": dict(Counter(row["capture_type"] for row in readable
                                             if row["label"] == "nothing")),
            "image_status": dict(Counter(row["image_status"] for row in samples)),
            "review_status_readable": dict(Counter(row["review_status"] for row in readable)),
            "exclusions": sum(bool(row["exclusions"]) for row in samples),
            "flags": dict(Counter(flag for row in readable for flag in row["flags"])),
        }

    external = {}
    for row in by_split["external_test"]:
        if row["image_status"] == "readable":
            external.setdefault(row["collection_group"], []).append(row)
    external_facts = {}
    for group, samples in external.items():
        exif_dates = sorted(row["exif_datetime_original"] for row in samples if row["exif_datetime_original"])
        external_facts[group] = {
            "readable": len(samples),
            "counts_by_target": {
                label if kind == "known_hand" else f"{label}/{kind}":
                sum((row["label"], row["capture_type"]) == (label, kind) for row in samples)
                for label, kind in TARGETS
            },
            "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in samples)),
            "exif_datetime_original_first": exif_dates[0] if exif_dates else None,
            "exif_datetime_original_last": exif_dates[-1] if exif_dates else None,
            "exif_datetime_original_count": len(exif_dates),
            "exif_offset_count": sum(bool(row["exif_offset_time_original"]) for row in samples),
            "exif_software": sorted({row["exif_software"] for row in samples if row["exif_software"]}),
        }

    fit = [row for split in FIT_SPLITS for row in by_split[split]]
    train_approved = counts.get("train", {}).get("reviewed_positives_by_class", {})
    missing_train_classes = [label for label in CLASSES_29 if not train_approved.get(label)]
    unresolved = [row for row in fit if not row["exclusions"] and row["image_status"] == "readable"
                  and (row["review_status"] != "approved" or "yolo_label" not in row)]
    near_pending = sum(group["touches_development"] and not group["review"]
                       for group in duplicates["perceptual_hash"]["cross_split_groups"])
    blockers = []
    if any(not counts.get(split, {}).get("with_yolo_label") for split in FIT_SPLITS):
        blockers.append("empty_train_or_validation")
    if missing_train_classes:
        blockers.append("missing_reviewed_positive_training_classes")
    if unresolved:
        blockers.append("pending_training_or_validation_annotations")
    if near_pending:
        blockers.append("pending_cross_split_near_duplicate_review")
    if any(row["image_status"] == "source_error" for row in rows):
        blockers.append("source_image_errors")
    if not {"no_hand", "other_hand"} <= set(counts.get("external_test", {}).get("nothing_subtypes", {})):
        blockers.append("missing_external_nothing_subtype")

    summary = {
        "training_ready": not blockers, "blockers": blockers, "split_counts": counts,
        "external_observations_from_images": external_facts,
        "missing_reviewed_train_classes": missing_train_classes,
        "development_other_hand_reviewed_train": train_approved.get("nothing", 0),
        "pending_train_val_annotations": len(unresolved),
        "pending_cross_split_near_duplicate_groups_touching_train_val": near_pending,
        "external_annotations_pending": sum(row["review_status"] == "pending"
                                            for row in by_split["external_test"]),
    }
    if any(row.get("review", {}).get("review_kind") == "automated" for row in fit):
        summary["fit_approvals_by_review_kind"] = dict(Counter(
            row["review"].get("review_kind", "human") for row in fit
            if row.get("yolo_label") and row["review_status"] == "approved"
        ))
    return summary


def verify_row_files(root, row):
    if row["image_status"] != "readable":
        return None, None
    image = Path(row["image"])
    sample_id = row["sample_id"]
    scope = "draft" if row["split"] in FIT_SPLITS else "holdout"
    if (image.parts[:3] != (scope, "images", row["split"]) or len(image.parts) != 4
            or image.stem != sample_id or image.suffix not in IMAGE_SUFFIXES
            or hashlib.sha256((root / image).read_bytes()).hexdigest() != row["output_sha256"]):
        raise ValueError(f"Invalid or altered image: {image}")
    if "yolo_label" not in row:
        if not row["exclusions"] and (row["boxes"] or not row["requires_box"]):
            raise ValueError(f"Missing label for {image}")
        return image, None
    label = Path(scope, "labels", row["split"], sample_id + ".txt")
    if row["yolo_label"] != label.as_posix() or row["exclusions"]:
        raise ValueError(f"Invalid label path or exclusion for {image}")
    boxes = []
    for line in (root / label).read_text(encoding="ascii").splitlines():
        class_id, *box = line.split()
        if class_id != str(row["class_id"]):
            raise ValueError(f"Incorrect object class in {label}")
        boxes.append([float(v) for v in box])
    validate_annotation(row["label"], row["capture_type"], boxes)
    if boxes != [[float(f"{v:.10f}") for v in box] for box in row["boxes"]]:
        raise ValueError(f"Emitted boxes differ from manifest: {label}")
    return image, label


def verify_files(root, expected_root, *, classes=CLASSES_29):
    """Check the emitted YOLO format, exact list membership and integrity on disk."""
    manifest = read_json(root / "manifest.json")
    if tuple(manifest["classes"]) != classes:
        raise ValueError("Unexpected manifest class order")
    class_ids = {label: index for index, label in enumerate(classes)}
    config = yaml.safe_load((root / "draft/data.yaml").read_text(encoding="utf-8"))
    expected_config = {"path": (expected_root / "draft").as_posix(), "train": "train.txt",
                       "val": "val.txt", "names": dict(enumerate(classes))}
    if config != expected_config:
        raise ValueError("Unexpected data.yaml paths or names")
    images, labels, sample_ids = set(), set(), set()
    lists = {split: [] for split in FIT_SPLITS}
    rows = manifest["rows"]
    for row in rows:
        spec = annotation_spec(row["label"], row["capture_type"])
        spec["class_id"] = class_ids[row["label"]]
        sample_id = row["sample_id"]
        if sample_id in sample_ids or any(row[key] != value for key, value in spec.items()):
            raise ValueError(f"Invalid or repeated manifest entry: {sample_id}")
        sample_ids.add(sample_id)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for row, (image, label) in zip(rows, pool.map(partial(verify_row_files, root), rows)):
            if image is not None:
                images.add(image)
            if label is not None:
                labels.add(label)
                if row["split"] in FIT_SPLITS:
                    if row.get("role") == "external_test" or row.get("person", "").casefold() == "bao":
                        raise ValueError(f"External capture in a development list: {image}")
                    lists[row["split"]].append("./" + image.relative_to("draft").as_posix())
    for split, paths in lists.items():
        if paths != (root / "draft" / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            raise ValueError(f"Unexpected image membership in {split}.txt")
    for folder, expected in (("images", images), ("labels", labels)):
        found = {Path(directory).relative_to(root) / filename for scope in ("draft", "holdout")
                 for directory, _subdirs, files in os.walk(root / scope / folder) for filename in files}
        if found != expected:
            raise ValueError(f"Unexpected {folder} inventory")
    return {"images_verified": len(images), "yolo_labels_verified": len(labels),
            "fit_lists_verified": {split: len(paths) for split, paths in lists.items()}}
