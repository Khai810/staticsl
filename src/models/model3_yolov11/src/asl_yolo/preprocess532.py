"""Image-level, provenance-preserving November ASL29 preprocessing.

Reuses dataset29's image decoding/box geometry, protocol validation and HandTracker.
This module deliberately has no model train/predict/evaluate entry point.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
from copy import deepcopy
from functools import partial
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml

from .core import digest, file_hash, read_json, write_json
from .dataset29 import git_blob, image_details, propose_box
from .protocol import CLASSES_29, load_protocol, validate_annotation
from .vision import HandTracker, yolo_to_xyxy

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "group_test")


def now():
    return datetime.now(timezone.utc).isoformat()


def log(output, step, **details):
    record = {"at_utc": now(), "step": step, **details}
    with (output / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(step + ": " + json.dumps(details, ensure_ascii=True), flush=True)


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                          else value for key, value in row.items()} for row in records)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def git(repository, *args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repository), *args],
                                   env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}, encoding="utf-8")


def source_tree(repo, commit):
    if git(repo, "rev-parse", "HEAD").strip() != commit:
        raise ValueError("Reference checkout is not at the protocol commit")
    tree = {}
    for entry in git(repo, "ls-tree", "-r", "-z", commit).rstrip("\0").split("\0"):
        info, name = entry.split("\t", 1)
        if info.split()[1] == "blob":
            tree[name] = info.split()[2]
    return tree


def canonical_label(name):
    if name in CLASSES_29:
        return name
    aliases = {"delete": "del", "no_hand": "nothing", "other_hand": "nothing"}
    return aliases.get(name.casefold(), name.upper() if name.upper() in CLASSES_29 else name)


def inventory_paths(root, protocol):
    repo = root / protocol["source"]["local_repository"]
    raw_root = Path(protocol["source"]["raw_label_root"])
    paths = []
    for path in sorted((repo / "data/raw").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        rel = path.relative_to(repo)
        label = canonical_label(path.parent.name)
        source, member, role, condition = "group_raw", None, "development", None
        exclusions = []
        if "custom_captured" in rel.parts:
            source = "group_capture"
            member = next((p.removesuffix("_dataset") for p in rel.parts if p.endswith("_dataset")), None)
            member = {m.casefold(): m for m in ("Khai", "Viet", "Bao", "Tien", "Dat")}.get(
                (member or "").casefold(), member)
            if member == "Bao":
                role = "external_test"
        elif raw_root.as_posix() not in rel.as_posix():
            source = "group_raw_alias"
            label = canonical_label(path.stem.removesuffix("_test"))
            exclusions.append("source_reference_alias_not_additional_sample")
        capture_type = "known_hand" if label != "nothing" else "unresolved"
        if source == "group_raw" and label == "nothing":
            capture_type = "no_hand"
        if source == "group_capture":
            for part in rel.parts:
                if part.casefold() in {"no_hand", "other_hand"}:
                    capture_type = part.casefold()
        paths.append(dict(source=source, source_path=rel.as_posix(), original_path=path.relative_to(root).as_posix(),
                          label=label, member=member, role=role, condition=condition,
                          session_id=None, burst_id=None, metadata_evidence=None, capture_type=capture_type,
                          exclusions=exclusions))
    inbox = root / protocol["source"]["inbox"]
    for path in sorted(inbox.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        rel = path.relative_to(inbox)
        role, member, condition = rel.parts[:3]
        label = canonical_label(rel.parts[3])
        capture_type = rel.parts[4] if label == "nothing" and len(rel.parts) > 5 else "known_hand"
        # phien_01 is an explicit session folder; dieu_kien_01 is a condition, not a session.
        session = "/".join(rel.parts[:3]) if condition.startswith("phien_") else None
        paths.append(dict(source="capture_inbox", source_path=rel.as_posix(), original_path=path.relative_to(root).as_posix(),
                          label=label, member=member.capitalize(), role=role, condition=condition,
                          session_id=session, burst_id=None, metadata_evidence="explicit inbox directory",
                          capture_type=capture_type, exclusions=[]))
    for row in paths:
        row.update(sample_id=digest(["staticsl_november" if row["source"] != "capture_inbox" else "capture_inbox",
                                    row["source_path"]])[:24], boxes=[], flags=[], annotation_state="pending",
                   annotation_reason="no_box_proposal", review_history=[], split_status="proposed",
                   split=None, image_status="unverified")
        if row["label"] not in CLASSES_29:
            row["exclusions"].append("invalid_label")
        row["class_id"] = CLASSES_29.index(row["label"]) if row["label"] in CLASSES_29 else None
    return paths


def inspect_image(root, tree, row):
    row = dict(row)
    try:
        content = (root / row["original_path"]).read_bytes()
        row["sha256"] = __import__("hashlib").sha256(content).hexdigest()
        if row["source"] != "capture_inbox":
            expected = tree.get(row["source_path"])
            row["git_blob_sha1"] = git_blob(content)
            if not expected or row["git_blob_sha1"] != expected:
                row["exclusions"].append("source_differs_from_pinned_git_blob")
        details, _rgb = image_details(content)
        row.update(details, image_status="readable")
    except Exception as exc:
        row.update(image_status="unreadable", image_error=f"{type(exc).__name__}: {exc}")
        row["exclusions"].append("image_unreadable")
    return row


def csv_key(record):
    member = record.get("member", "")
    if member.casefold() == "nan":
        member = ""
    return (member, record["label"], record["source_file"])


def row_key(row):
    return (row["member"] or "", row["label"], Path(row["source_path"]).name)


def audit_group_csv(root, repo, output, tree, rows):
    """Re-run the group's own splitter into a new audit folder, preserving its sources."""
    csv_files = sorted((repo / "data/hybrid_manifest").glob("*.csv")) + sorted((repo / "data/landmarks").glob("*.csv"))
    hashes = {}
    for path in csv_files:
        rel = path.relative_to(repo).as_posix()
        content = path.read_bytes()
        blob = git_blob(content)
        canonical_blob = git_blob(content.replace(b"\r\n", b"\n"))
        if tree.get(rel) not in {blob, canonical_blob}:
            raise ValueError(f"Unpinned source CSV: {rel}")
        hashes[rel] = {"sha256": file_hash(path), "worktree_blob_sha1": blob,
                       "git_blob_sha1": tree[rel],
                       "git_eol_normalization": "CRLF to LF" if blob != tree[rel] else "none"}
    generated = output / "audit/group_regenerated_csv"
    generated.mkdir(parents=True, exist_ok=True)
    with (output / "audit/group_regeneration.log").open("w", encoding="utf-8") as stream:
        subprocess.run([str(Path(os.sys.executable)), str(repo / "src/preprocessing/create_manifest_5_3_2.py"),
                        "--raw_csv", str(repo / "data/landmarks/landmarks_raw.csv"),
                        "--cap_csv", str(repo / "data/landmarks/landmarks_captured.csv"),
                        "--output_dir", str(generated), "--seed", "42"], check=True, stdout=stream,
                       stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    by_key = defaultdict(list)
    for row in rows:
        if row["source"] in {"group_raw", "group_capture"}:
            by_key[row_key(row)].append(row)
    comparison, counts, regenerated, differences, conflicts = [], {}, {}, {}, []
    for split in ("train", "val", "test"):
        path = repo / f"data/hybrid_manifest/{split}.csv"
        current_keys, regenerated_keys = [], []
        for current, csv_path in ((True, path), (False, generated / path.name)):
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                for number, record in enumerate(csv.DictReader(stream), 2):
                    key = csv_key(record)
                    (current_keys if current else regenerated_keys).append(key)
                    if not current:
                        continue
                    matches = by_key[key]
                    item = dict(source_split=split, csv_row=number, member=key[0], label=key[1], source_file=key[2],
                                sample_ids=[r["sample_id"] for r in matches], mapping_status="matched" if len(matches) == 1 else "unresolved")
                    comparison.append(item)
                    for row in matches:
                        row.setdefault("source_csv_membership", []).append({"split": split, "row": number})
                        if row["role"] == "external_test":
                            conflicts.append({"kind": "group_csv_uses_reserved_external", **item})
            if file_hash(path) != hashes[path.relative_to(repo).as_posix()]["sha256"]:
                raise ValueError("Group CSV changed during audit")
        counts[split], regenerated[split] = len(current_keys), len(regenerated_keys)
        differences[split] = {"missing_from_committed": len(set(regenerated_keys) - set(current_keys)),
                              "extra_in_committed": len(set(current_keys) - set(regenerated_keys)),
                              "committed_duplicate_keys": len(current_keys) - len(set(current_keys))}
    write_csv(output / "audit/group_csv_membership.csv", comparison,
              ["source_split", "csv_row", "member", "label", "source_file", "sample_ids", "mapping_status"])
    write_jsonl(output / "audit/group_source_conflicts.jsonl", conflicts)
    result = dict(csv_hashes=hashes, committed_counts=counts, regenerated_counts=regenerated, differences=differences,
                  group_split_verified_official=False, reason="No signed/frozen official split; committed/recreated membership differs",
                  mapping_status=dict(Counter(r["mapping_status"] for r in comparison)),
                  group_csv_external_conflicts=len(conflicts))
    write_json(output / "audit/group_manifest_audit.json", result)
    return result


class Components:
    def __init__(self, count):
        self.parents = list(range(count))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, i, j):
        self.parents[self.find(j)] = self.find(i)


def exact_components(rows):
    components, seen = Components(len(rows)), {}
    for i, row in enumerate(rows):
        for field in ("sha256", "pixel_sha256"):
            if not row.get(field):
                continue
            key = field, row[field]
            if key in seen:
                components.union(i, seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[components.find(i)].append(rows[i])
    for group in groups.values():
        group_id = min(r["sample_id"] for r in group)
        external = any(r["role"] == "external_test" for r in group)
        condition = {r["condition"] for r in group if r["source"] == "capture_inbox" and r["condition"]}
        has_external_original = any(r["source"] == "capture_inbox" and r["role"] == "external_test" for r in group)
        known_types = {r["capture_type"] for r in group if r["capture_type"] != "unresolved"}
        if len(known_types) == 1 and len({r["label"] for r in group}) == 1:
            for row in group:
                if row["capture_type"] == "unresolved":
                    row["capture_type"] = next(iter(known_types))
                    row["capture_type_evidence"] = "exact/pixel hash match to explicitly typed original"
        conflict = len({(r["label"], r["capture_type"]) for r in group}) > 1
        for row in group:
            row["duplicate_group"] = group_id
            if external:
                if row["role"] != "external_test":
                    row["flags"].append("exact_copy_of_external_test")
                row["role"] = "external_test"
                if has_external_original and row["source"] != "capture_inbox":
                    row["exclusions"].append("confirmed_external_alias_use_inbox_original")
                if row["source"] != "capture_inbox" and len(condition) == 1:
                    row["condition"] = next(iter(condition))
                    row["metadata_evidence"] = "exact/pixel match to condition-specific inbox image"
            if conflict:
                row["flags"].append("exact_duplicate_label_or_capture_type_conflict")
                row["exclusions"].append("duplicate_label_conflict_pending_review")
    return groups


def largest_remainder(n, ratios=(.5, .3, .2)):
    exact = [n * r for r in ratios]
    result = [math.floor(v) for v in exact]
    for i in sorted(range(len(ratios)), key=lambda i: (-(exact[i] - result[i]), i))[:n - sum(result)]:
        result[i] += 1
    return result


def assign_proposed_splits(rows, seed=42):
    """Assign image units, merging exact copies and explicitly evidenced sessions/bursts."""
    comp, seen = Components(len(rows)), {}
    for i, row in enumerate(rows):
        for field in ("duplicate_group", "session_id", "burst_id"):
            if row.get(field):
                key = field, row[field]
                if key in seen:
                    comp.union(i, seen[key])
                seen[key] = i
    units = defaultdict(list)
    for i, row in enumerate(rows):
        units[comp.find(i)].append(row)
    strata = defaultdict(list)
    for unit in units.values():
        group_id = min(r["sample_id"] for r in unit)
        for row in unit:
            row["split_group"] = group_id
        if any(r["role"] == "external_test" for r in unit):
            for row in unit:
                row.update(split="external_test", role="external_test")
            continue
        eligible = [r for r in unit if not r["exclusions"]]
        if not eligible:
            for row in unit:
                row["split"] = "excluded"
            continue
        # A session spanning several labels is one indivisible unit, not a per-label split.
        key = tuple(sorted({("capture" if r["member"] else "raw", r["member"] or "", r["label"]) for r in eligible}))
        strata[key].append(unit)
    for key, units_in_stratum in sorted(strata.items()):
        units_in_stratum.sort(key=lambda group: group[0]["split_group"])
        random.Random(digest([seed, key])).shuffle(units_in_stratum)
        targets = largest_remainder(sum(sum(not r["exclusions"] for r in u) for u in units_in_stratum))
        assigned = [0, 0, 0]
        # Larger groups first: preserve independence even if tiny strata cannot achieve 5/3/2.
        units_in_stratum.sort(key=lambda unit: -len(unit))
        for unit in units_in_stratum:
            size = sum(not r["exclusions"] for r in unit)
            chosen = max(range(3), key=lambda i: (targets[i] - assigned[i], -i))
            assigned[chosen] += size
            for row in unit:
                row["split"] = SPLITS[chosen]
    return rows


def phash_candidates(rows, distance=2):
    """Exact 64-bit Hamming radius search using d+1 disjoint pigeonhole bands.

    Output is a direct pair flag, never an asserted duplicate component.
    """
    bands, candidates = defaultdict(list), []
    widths = [64 // (distance + 1) + (i < 64 % (distance + 1)) for i in range(distance + 1)]
    for i, row in enumerate(rows):
        if not row.get("phash"):
            continue
        value = int(row["phash"], 16)
        offset, prior = 0, set()
        keys = []
        for band, width in enumerate(widths):
            key = band, (value >> offset) & ((1 << width) - 1)
            keys.append(key)
            prior.update(bands[key])
            offset += width
        for j in sorted(prior):
            other = rows[j]
            if row["duplicate_group"] == other["duplicate_group"]:
                continue
            d = (value ^ int(other["phash"], 16)).bit_count()
            if d <= distance:
                candidates.append(dict(a=row["sample_id"], b=other["sample_id"], distance=d,
                                       cross_split=row["split"] != other["split"],
                                       includes_external="external_test" in (row["split"], other["split"]),
                                       status="pending_identity_review"))
        for key in keys:
            bands[key].append(i)
    return candidates


def isolate_candidates(rows, candidates):
    by_id = {r["sample_id"]: r for r in rows}
    for pair in candidates:
        if not pair["cross_split"]:
            continue
        # Preserve all holdout membership/images. Exclude development candidates from fit lists only.
        for sample_id in (pair["a"], pair["b"]):
            row = by_id[sample_id]
            if row["split"] in {"train", "val"}:
                reason = "phash_external_candidate_pending_review" if pair["includes_external"] else "phash_cross_split_candidate_pending_review"
                if reason not in row["exclusions"]:
                    row["exclusions"].append(reason)
    return rows


def reuse_annotations(root, protocol, rows):
    base = root / protocol["source"]["reviewed_snapshot"]
    snapshot = read_json(base / "manifest.json")
    prior = defaultdict(list)
    for old in snapshot["rows"]:
        if old.get("sha256"):
            prior[(old["sha256"], old["label"])].append(old)
    reused = Counter()
    for row in rows:
        matches = prior.get((row.get("sha256"), row["label"]), [])
        matches.sort(key=lambda r: {"approved": 0, "rejected": 1, "pending": 2}[r["review_status"]])
        if not matches or row["image_status"] != "readable":
            continue
        old = matches[0]
        row["flags"].extend(old.get("flags", []))
        if len(old.get("boxes", [])) > 1 or any("multiple_hands" in flag for flag in row["flags"]):
            row["flags"].append("multiple_hands")
            row["annotation_origin"] = dict(kind="hash_matched_previous_snapshot", manifest=protocol["source"]["reviewed_snapshot"] + "/manifest.json",
                                             old_sample_id=old["sample_id"], image_sha256=row["sha256"])
            row.setdefault("proposal_history", []).append({"boxes": old["boxes"], "reason": "prior multiple-hand proposal requires per-object labels"})
            row.update(annotation_state="pending", annotation_reason="multiple_hands_require_per_object_label_review")
            continue
        if row["capture_type"] == "unresolved":
            row["capture_type"] = old["capture_type"]
        if row["capture_type"] != old["capture_type"]:
            row["flags"].append("prior_capture_type_conflict")
            continue
        row["annotation_origin"] = dict(kind="hash_matched_previous_snapshot", manifest=protocol["source"]["reviewed_snapshot"] + "/manifest.json",
                                        old_sample_id=old["sample_id"], image_sha256=row["sha256"])
        review = old.get("review")
        if review:
            row["review_history"].append({**review, "at_utc": review.get("at_utc") or review.get("reviewed_at") or review.get("timestamp"),
                                          "timestamp_note": "unknown if absent in original; never fabricated"})
        if old["review_status"] == "rejected":
            row.update(annotation_state="pending", annotation_reason="prior_annotation_rejected_requires_new_review")
            row["flags"].append("prior_annotation_rejected")
        elif old["boxes"] or row["capture_type"] == "no_hand":
            try:
                validate_annotation(row["label"], row["capture_type"], old["boxes"])
                row.update(boxes=old["boxes"], annotation_state="reviewed" if review and old["review_status"] == "approved" else "proposed",
                           annotation_reason="reused_" + old["review_status"], review_kind=review.get("review_kind", "unknown") if review else None)
                reused[row["annotation_state"]] += 1
            except ValueError:
                row["flags"].append("invalid_previous_box")
    return dict(reused)


def add_csv_proposals(repo, tree, rows):
    by_key = defaultdict(list)
    for row in rows:
        if row["source"] in {"group_raw", "group_capture"}:
            by_key[row_key(row)].append(row)
    for name in ("raw", "captured"):
        path = repo / f"data/landmarks/landmarks_{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for number, record in enumerate(csv.DictReader(stream), 2):
                matches = by_key[csv_key(record)]
                if len(matches) != 1:
                    continue
                row = matches[0]
                row["source_landmark_csv_row"] = number
                if row["annotation_state"] != "pending" or row["image_status"] != "readable" or row["capture_type"] != "known_hand":
                    continue
                if "prior_annotation_rejected" in row["flags"] or "multiple_hands" in row["flags"] or row.get("exif_orientation") not in {1, None}:
                    continue
                try:
                    points = np.array([float(record[f"r{i}"]) for i in range(63)]).reshape(21, 3)
                    propose_box(row, points)
                    row.update(annotation_state="proposed", annotation_reason="pinned_raw_landmark_box_requires_review",
                               annotation_origin={"kind": "group_raw_landmark_csv", "csv": path.relative_to(repo).as_posix(),
                                                  "row": number, "csv_git_blob": tree[path.relative_to(repo).as_posix()],
                                                  "image_git_blob": row["git_blob_sha1"], "image_sha256": row["sha256"],
                                                  "padding_ratio": .15, "extractor": "group MediaPipe; original version not recorded",
                                                  "coordinate_space": "original full image; captured adaptive borders mapped back by group script"})
                except (ValueError, KeyError) as exc:
                    row["annotation_reason"] = "invalid_source_landmarks: " + str(exc)


def map_padded_points(points, width, height, border):
    points = np.asarray(points, dtype=float).copy()
    points[:, 0] = (points[:, 0] * (width + 2 * border) - border) / width
    points[:, 1] = (points[:, 1] * (height + 2 * border) - border) / height
    points[:, 2] *= (width + 2 * border) / width
    return points


def propose_capture_boxes(root, protocol, rows):
    import cv2
    version = importlib.metadata.version("mediapipe")
    asset = root / "assets/hand_landmarker.task"
    trials = protocol["annotation"]["border_trials"]
    with HandTracker(asset) as tracker:
        for row in rows:
            if row["source"] not in {"capture_inbox", "group_capture"} or row["image_status"] != "readable":
                continue
            if (row["annotation_state"] != "pending" or row["capture_type"] != "known_hand"
                    or "prior_annotation_rejected" in row["flags"] or "multiple_hands" in row["flags"]):
                continue
            with Image.open(root / row["original_path"]) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            row["mediapipe_attempts"] = []
            for trial in trials:
                border = trial["pixels"]
                frame = cv2.copyMakeBorder(rgb, border, border, border, border,
                                            cv2.BORDER_REPLICATE if trial["mode"] == "replicate" else cv2.BORDER_CONSTANT)
                hands = tracker.detect(frame)
                row["mediapipe_attempts"].append({**trial, "hand_count": len(hands)})
                if len(hands) > 1:
                    row.update(annotation_reason="multiple_hands_require_per_object_label_review")
                    row["flags"].append("multiple_hands")
                    break
                if len(hands) == 1:
                    try:
                        points = map_padded_points(hands[0], row["width"], row["height"], border)
                        propose_box(row, points)
                        row.update(annotation_state="proposed", annotation_reason="mediapipe_proposal_requires_review",
                                   annotation_origin={"kind": "MediaPipe HandLandmarker", "version": version,
                                                      "asset_sha256": file_hash(asset), "padding_ratio": .15,
                                                      "border_trial": trial, "mapped_to_original": True})
                    except ValueError as exc:
                        row["annotation_reason"] = "invalid_mediapipe_box: " + str(exc)
                    break
            else:
                row["annotation_reason"] = "mediapipe_failed_still_pending_hand_annotation"


def flag_annotations(rows):
    for row in rows:
        if row["capture_type"] == "unresolved":
            row.update(boxes=[], annotation_state="pending", annotation_reason="nothing_capture_type_not_confirmed")
        if len(row["boxes"]) > 1:
            row["flags"].append("multiple_hands")
        if "multiple_hands" in row["flags"]:
            if row["boxes"]:
                row.setdefault("proposal_history", []).append({"boxes": row["boxes"], "reason": "multiple hands; folder label cannot be assigned to every hand"})
            row.update(boxes=[], annotation_state="pending", annotation_reason="multiple_hands_require_per_object_label_review")
        for box in row["boxes"]:
            try:
                validate_annotation(row["label"], row["capture_type"], [box])
            except ValueError as exc:
                row.update(annotation_state="pending", annotation_reason="invalid_box: " + str(exc))
                row["boxes"] = []
                break
            x, y, w, h = box
            if min(x - w / 2, y - h / 2, 1 - x - w / 2, 1 - y - h / 2) < 1e-6:
                row["flags"].append("box_touches_image_edge")
            if w * h < .01 or w * h > .85:
                row["flags"].append("unusual_box_area")
            if w / h < .15 or w / h > 6:
                row["flags"].append("unusual_box_aspect_ratio")
        row["flags"] = sorted(set(row["flags"]))
        row["exclusions"] = sorted(set(row["exclusions"]))


def export_one(root, output, row):
    if row["image_status"] != "readable" or row["split"] == "excluded":
        return
    scope = "draft" if row["split"] in {"train", "val"} else "holdout"
    extension = ".jpg" if row["format"] == "JPEG" else ".png"
    oriented = row["exif_orientation"] not in {1, None}
    if oriented:
        extension = ".png"
    image = Path(scope, "images", row["split"], row["sample_id"] + extension)
    target = output / image
    target.parent.mkdir(parents=True, exist_ok=True)
    if oriented:
        with Image.open(root / row["original_path"]) as original:
            ImageOps.exif_transpose(original).convert("RGB").save(target)
    else:
        # Independent copies protect original data even if a future loader repairs an image.
        shutil.copyfile(root / row["original_path"], target)
    row.update(image=image.as_posix(), output_sha256=file_hash(target),
               export_conversion="EXIF-transposed lossless PNG" if oriented else "byte-for-byte copy; extension from actual format")
    has_annotation = bool(row["boxes"]) or (row["capture_type"] == "no_hand" and row["annotation_state"] == "reviewed")
    if has_annotation and row["annotation_state"] in {"proposed", "reviewed"}:
        validate_annotation(row["label"], row["capture_type"], row["boxes"])
        label = Path(scope, "labels", row["split"], row["sample_id"] + ".txt")
        (output / label).parent.mkdir(parents=True, exist_ok=True)
        content = "".join(str(row["class_id"]) + " " + " ".join(f"{v:.10f}" for v in box) + "\n" for box in row["boxes"])
        (output / label).write_text(content, encoding="ascii")
        row.update(yolo_label=label.as_posix(), label_sha256=file_hash(output / label))
    else:
        # Only remove this build's own stale draft label when resuming unfinished work.
        (output / scope / "labels" / row["split"] / (row["sample_id"] + ".txt")).unlink(missing_ok=True)
    row["exported_for_loader"] = bool(row.get("yolo_label")) and not row["exclusions"]


def write_lists_and_configs(output, rows):
    names = dict(enumerate(CLASSES_29))
    for scope in ("draft", "holdout"):
        for folder in ("images", "labels"):
            (output / scope / folder).mkdir(parents=True, exist_ok=True)
    for split in (*SPLITS, "external_test"):
        scope = "draft" if split in {"train", "val"} else "holdout"
        folder = output / scope
        subset = [r for r in rows if r["split"] == split and r.get("image")]
        for suffix, selection in (("_all", subset), ("", [r for r in subset if r["exported_for_loader"]])):
            (folder / (split + suffix + ".txt")).write_text("".join("./" + Path(r["image"]).relative_to(scope).as_posix() + "\n" for r in selection), encoding="utf-8")
        if split == "external_test":
            canonical = [r for r in subset if "confirmed_external_alias_use_inbox_original" not in r["exclusions"]]
            (folder / "external_test_canonical_all.txt").write_text("".join("./" + Path(r["image"]).relative_to(scope).as_posix() + "\n" for r in canonical), encoding="utf-8")
    for condition in sorted({r["condition"] for r in rows if r["split"] == "external_test" and r["condition"]}):
        subset = [r for r in rows if r["split"] == "external_test" and r["condition"] == condition and r.get("image")]
        (output / "holdout" / f"external_{condition}_all.txt").write_text("".join("./" + Path(r["image"]).relative_to("holdout").as_posix() + "\n" for r in subset), encoding="utf-8")
    configs = {"draft/data.yaml": {"path": (output / "draft").as_posix(), "train": "train.txt", "val": "val.txt", "names": names}}
    for split in ("group_test", "external_test"):
        configs[f"holdout/{split}.yaml"] = {"path": (output / "holdout").as_posix(), "train": f"{split}.txt", "val": f"{split}.txt", "test": f"{split}.txt", "names": names}
    for path, config in configs.items():
        (output / path).write_text("# DRAFT: annotated subset only; incomplete ground truth is a blocker. No evaluation authorized.\n" + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def manifest_fingerprint(rows):
    return digest([{key: row.get(key) for key in ("sample_id", "source", "source_path", "sha256", "pixel_sha256", "label", "class_id", "capture_type", "role", "condition", "session_id", "burst_id", "split", "split_group", "exclusions", "flags", "annotation_state", "annotation_origin", "review_history", "boxes", "image", "output_sha256", "yolo_label", "label_sha256", "exported_for_loader")}
                   for row in rows])


def final_draft_qc(output, root=None):
    """Finish QC of an unfrozen draft before handover; keep changes in the step log."""
    manifest = read_json(output / "manifest.json")
    if manifest.get("split_frozen"):
        raise ValueError("Final QC cannot modify a frozen dataset")
    rows = read_jsonl(output / "manifest.jsonl")
    before = digest(rows)
    if root is not None:
        protocol = read_json(output / "protocol.json")
        prior = defaultdict(list)
        for old in read_json(root / protocol["source"]["reviewed_snapshot"] / "manifest.json")["rows"]:
            if old.get("sha256"):
                prior[(old["sha256"], old["label"])].append(old)
        for row in rows:
            matches = prior.get((row.get("sha256"), row["label"]), [])
            if matches:
                matches.sort(key=lambda r: {"approved": 0, "rejected": 1, "pending": 2}[r["review_status"]])
                row["flags"].extend(matches[0].get("flags", []))
    exact_components(rows)
    flag_annotations(rows)
    for row in rows:
        if row.get("yolo_label") and not row["boxes"] and row["capture_type"] != "no_hand":
            path = output / row["yolo_label"]
            scope = "draft" if row["split"] in {"train", "val"} else "holdout"
            expected = output / scope / "labels" / row["split"] / (row["sample_id"] + ".txt")
            if path.resolve() != expected.resolve():
                raise ValueError("Unexpected label location during final holdout QC")
            path.unlink(missing_ok=True)
            row.pop("yolo_label")
            row.pop("label_sha256")
            row["exported_for_loader"] = False
        if row.get("image"):
            row["exported_for_loader"] = bool(row.get("yolo_label")) and not row["exclusions"]
    fingerprint = manifest_fingerprint(rows)
    if before == digest(rows) and fingerprint == manifest["fingerprint"]:
        return
    prior = manifest["fingerprint"]
    manifest["fingerprint"] = fingerprint
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "manifest.json", manifest)
    write_lists_and_configs(output, rows)
    fields = next(csv.reader((output / "manifest.csv").open(encoding="utf-8")))
    write_csv(output / "manifest.csv", rows, fields)
    for split in (*SPLITS, "external_test", "excluded"):
        write_csv(output / "splits" / f"{split}.csv", [r for r in rows if r["split"] == split], fields)
    write_csv(output / "audit/review_queue.csv", [r for r in rows if r["annotation_state"] != "reviewed" or r["exclusions"]], fields)
    report = read_json(output / "audit/report.json")
    report.update(summarize(rows))
    if not report.get("technical_checks_passed"):
        raise ValueError("A complete checksum verification is required before metadata-only final QC")
    report["prior_full_checksum_verification"] = {key: report[key] for key in ("technical_checks_passed", "images_verified", "labels_verified", "fingerprint")}
    report.update(verify(output, verify_checksums=False))
    write_json(output / "audit/report.json", report)
    state = read_json(output / "build_state.json")
    state["fingerprint"] = fingerprint
    write_json(output / "build_state.json", state)
    log(output, "final_draft_qc", previous_fingerprint=prior, fingerprint=fingerprint,
        policy="multiple-hand proposals remain pending; one folder label cannot annotate every hand")


def augmentation_config(protocol, mirror=False):
    config = dict(protocol["augmentation"])
    for key in ("status", "mirror_variant_fliplr", "imgsz"):
        config.pop(key)
    config["fliplr"] = .5 if mirror else 0.
    return config


def make_loader(output, split, config, augment):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    scope = "draft" if split in {"train", "val"} else "holdout"
    # augmentations=[] explicitly suppresses optional Albumentations defaults in the pinned version.
    hyp = get_cfg(overrides={k: v for k, v in config.items() if k != "augmentations"})
    hyp.augmentations = config.get("augmentations", [])
    return YOLODataset(img_path=str(output / scope / f"{split}.txt"), imgsz=320, batch_size=4,
                       augment=augment, hyp=hyp, rect=False, cache=False, stride=32,
                       data={"names": dict(enumerate(CLASSES_29)), "nc": 29, "channels": 3}, task="detect")


def draw_tile(image, boxes, title, color="lime"):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(yolo_to_xyxy(box, *image.size), outline=color, width=max(2, image.width // 150))
    tile = Image.new("RGB", (240, 200), "#20252b")
    tile.paste(ImageOps.contain(image, (240, 165)), (0, 30))
    ImageDraw.Draw(tile).text((4, 3), title[:38], fill="white")
    return tile


def save_sheet(path, tiles, columns=5):
    if not tiles:
        return
    sheet = Image.new("RGB", (columns * 240, math.ceil(len(tiles) / columns) * 200), "#111820")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 240, (i // columns) * 200))
    sheet.save(path)


def transform_summary(transform):
    """Stable effective parameters, excluding memory addresses and dataset references."""
    result = {"type": type(transform).__name__}
    for field in ("p", "degrees", "translate", "scale", "shear", "perspective", "border",
                  "hgain", "sgain", "vgain", "direction", "bgr", "normalize", "bbox_format",
                  "new_shape", "auto", "scale_fill", "scaleup", "center", "stride"):
        value = getattr(transform, field, None)
        if isinstance(value, (str, int, float, bool, list, tuple)):
            result[field] = value
    if hasattr(transform, "transforms"):
        result["transforms"] = [transform_summary(t) for t in transform.transforms]
    if type(transform).__name__ == "Albumentations":
        result["effective_transform"] = str(transform.transform) if transform.transform is not None else None
    if getattr(transform, "pre_transform", None) is not None:
        result["pre_transform"] = transform_summary(transform.pre_transform)
    return result


def previews_and_loader(output, protocol, rows):
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    preview = output / "previews"
    preview.mkdir(exist_ok=True)
    selections = {}
    # Cover source, class, split and condition without choosing augmentation from holdout.
    for dimension in ("source", "label", "split", "condition"):
        selected = []
        seen = set()
        for row in rows:
            value = row.get(dimension)
            if row.get("image") and value not in seen:
                selected.append(row)
                seen.add(value)
        tiles = []
        for row in selected:
            with Image.open(output / row["image"]) as image:
                tiles.append(draw_tile(image, row["boxes"], f"{row['label']} {str(row.get(dimension))} {row['annotation_state']}"))
        save_sheet(preview / (dimension + ".jpg"), tiles)
        selections[dimension] = [r["sample_id"] for r in selected]
    results = {}
    for mirror in (False, True):
        name = "mirror" if mirror else "baseline"
        config = augmentation_config(protocol, mirror)
        (output / f"augmentation_{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        dataset = make_loader(output, "train", config, True)
        batch = dataset.collate_fn([dataset[i] for i in range(min(4, len(dataset)))])
        tiles = []
        selected_ids = []
        indices = []
        for label in CLASSES_29:
            idx = next((i for i, entry in enumerate(dataset.labels) if label in [CLASSES_29[int(v)] for v in entry["cls"].flatten()]), None)
            if idx is not None:
                indices.append(idx)
        indices = indices[:8]
        by_id = {r["sample_id"]: r for r in rows}
        for selector in (lambda r: r["source"] == "group_capture",
                         lambda r: r["source"] == "capture_inbox",
                         lambda r: r["capture_type"] == "no_hand",
                         lambda r: r["class_id"] == 27 and bool(r["boxes"])):
            idx = next((i for i, entry in enumerate(dataset.labels) if selector(by_id[Path(entry["im_file"]).stem])), None)
            if idx is not None and idx not in indices:
                indices.append(idx)
        for i in indices:
            result = dataset[i]
            rgb = result["img"].numpy().transpose(1, 2, 0)
            boxes = result["bboxes"].numpy().tolist()
            tiles.append(draw_tile(Image.fromarray(rgb), boxes, f"{name} {Path(result['im_file']).stem[:12]}"))
            selected_ids.append(Path(result["im_file"]).stem)
        save_sheet(preview / f"augmentation_{name}.jpg", tiles, 4)
        results[name] = {"images": len(dataset), "batch_shape": list(batch["img"].shape),
                         "boxes_shape": list(batch["bboxes"].shape), "transforms": str(dataset.transforms),
                         "effective_transforms": transform_summary(dataset.transforms),
                         "config": config, "preview_samples": selected_ids}
    validation = make_loader(output, "val", augmentation_config(protocol), False)
    batch = validation.collate_fn([validation[i] for i in range(min(4, len(validation)))])
    results["val"] = {"images": len(validation), "batch_shape": list(batch["img"].shape), "transforms": str(validation.transforms),
                      "effective_transforms": transform_summary(validation.transforms)}
    results["preview_samples"] = selections
    write_json(output / "audit/loader_and_augmentation.json", results)
    return results


def summarize(rows):
    distributions = {}
    for dimension in ("source", "label", "member", "condition", "split", "annotation_state", "capture_type", "review_kind"):
        distributions[dimension] = dict(Counter(str(r.get(dimension)) for r in rows))
    split_summary = {}
    for split in (*SPLITS, "external_test", "excluded"):
        subset = [r for r in rows if r["split"] == split]
        split_summary[split] = dict(total=len(subset), readable=sum(r["image_status"] == "readable" for r in subset),
                                    image_exports=sum(bool(r.get("image")) for r in subset),
                                    annotated=sum(bool(r.get("yolo_label")) for r in subset),
                                    loader_subset=sum(bool(r.get("exported_for_loader")) for r in subset),
                                    reviewed=sum(r["annotation_state"] == "reviewed" for r in subset),
                                    pending=sum(r["annotation_state"] == "pending" for r in subset),
                                    background=sum(r["capture_type"] == "no_hand" and bool(r.get("yolo_label")) for r in subset),
                                    positive_nothing=sum(r["class_id"] == 27 and bool(r["boxes"]) for r in subset),
                                    class_positive_boxes=dict(Counter(r["label"] for r in subset if r.get("exported_for_loader") for _ in r["boxes"])))
        readable = split_summary[split]["readable"]
        split_summary[split]["annotation_coverage"] = split_summary[split]["annotated"] / readable if readable else 0.
        split_summary[split]["reviewed_coverage"] = split_summary[split]["reviewed"] / readable if readable else 0.
    return dict(total_images=len(rows), unreadable=sum(r["image_status"] != "readable" for r in rows),
                distributions=distributions, splits=split_summary,
                exclusions=dict(Counter(reason for r in rows for reason in r["exclusions"])),
                flags=dict(Counter(reason for r in rows for reason in r["flags"])))


def verify(output, *, verify_checksums=True):
    manifest = read_json(output / "manifest.json")
    rows = read_jsonl(output / "manifest.jsonl")
    if manifest["fingerprint"] != manifest_fingerprint(rows) or tuple(manifest["classes"]) != CLASSES_29:
        raise ValueError("Manifest fingerprint/class mapping mismatch")
    replay = deepcopy(read_jsonl(output / "audit/inventory_cache.jsonl"))
    exact_components(replay)
    assign_proposed_splits(replay, read_json(output / "protocol.json")["split_policy"]["seed"])
    if {r["sample_id"]: r["split"] for r in replay} != {r["sample_id"]: r["split"] for r in rows}:
        raise ValueError("Seed-42 split replay differs from manifest membership")
    seen, groups, images, labels = set(), defaultdict(set), set(), set()
    def check_row(row):
        if not row.get("image"):
            return
        path = output / row["image"]
        if path.stem != row["sample_id"] or file_hash(path) != row["output_sha256"]:
            raise ValueError(f"Image checksum/path mismatch: {path}")
        if row.get("yolo_label"):
            path = output / row["yolo_label"]
            if file_hash(path) != row["label_sha256"]:
                raise ValueError(f"Label checksum mismatch: {path}")
            boxes = []
            for line in path.read_text(encoding="ascii").splitlines():
                values = line.split()
                if len(values) != 5 or values[0] != str(row["class_id"]):
                    raise ValueError(f"Invalid YOLO object line: {path}")
                boxes.append(list(map(float, values[1:])))
            validate_annotation(row["label"], row["capture_type"], boxes)
            if boxes != [[float(f"{v:.10f}") for v in box] for box in row["boxes"]]:
                raise ValueError("Serialized boxes differ from the manifest")
    if verify_checksums:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for _ in pool.map(check_row, rows):
                pass
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["sample_id"])
        if row.get("image"):
            images.add(row["image"])
        if row.get("yolo_label"):
            labels.add(row["yolo_label"])
        if row["role"] == "external_test" and row["split"] != "external_test":
            raise ValueError("External sample assigned to development")
        if row["split"] != "excluded":
            groups[row["split_group"]].add(row["split"])
        if row.get("exported_for_loader") and row["exclusions"]:
            raise ValueError("Quarantined sample in a loader list")
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Exact duplicate/session/burst crosses splits")
    for folder, expected in (("images", images), ("labels", labels)):
        found = {(Path(directory).relative_to(output) / filename).as_posix()
                 for scope in ("draft", "holdout")
                 for directory, _subdirs, files in os.walk(output / scope / folder)
                 for filename in files if not filename.endswith(".cache")}
        if found != expected:
            raise ValueError(f"Unexpected or missing exported {folder}")
    for split in (*SPLITS, "external_test"):
        scope = "draft" if split in {"train", "val"} else "holdout"
        for suffix, predicate in (("_all", lambda r: bool(r.get("image"))), ("", lambda r: bool(r.get("exported_for_loader")))):
            expected = ["./" + Path(r["image"]).relative_to(scope).as_posix() for r in rows if r["split"] == split and predicate(r)]
            if (output / scope / f"{split}{suffix}.txt").read_text(encoding="utf-8").splitlines() != expected:
                raise ValueError(f"Image list membership mismatch: {split}{suffix}")
        cfg_path = output / ("draft/data.yaml" if scope == "draft" else f"holdout/{split}.yaml")
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if config["names"] != dict(enumerate(CLASSES_29)) or config["path"] != (output / scope).as_posix():
            raise ValueError("YAML path/class mapping mismatch; use --relocate after copying")
        if scope == "draft" and (config.get("train") != "train.txt" or config.get("val") != "val.txt" or config.get("test")):
            raise ValueError("Holdout or unexpected path in draft config")
    candidates = read_jsonl(output / "audit/phash_candidates.jsonl")
    active = {r["sample_id"] for r in rows if r.get("exported_for_loader") and r["split"] in {"train", "val"}}
    if any(p["cross_split"] and (p["a"] in active or p["b"] in active) for p in candidates):
        raise ValueError("Unresolved cross-split pHash candidate in fit lists")
    return dict(technical_checks_passed=True, images_verified=len(images), labels_verified=len(labels),
                fingerprint=manifest["fingerprint"], training_ready=False,
                verification_mode="full_checksums" if verify_checksums else "metadata_after_prior_full_checksum_verification")


def relocate(output):
    rows = read_jsonl(output / "manifest.jsonl")
    if manifest_fingerprint(rows) != read_json(output / "manifest.json")["fingerprint"]:
        raise ValueError("Fingerprint differs before relocation")
    write_lists_and_configs(output, rows)
    return verify(output)


def build(root, protocol_path, output, resume=False):
    protocol = load_protocol(protocol_path)
    state = output / "build_state.json"
    if output.exists():
        if not resume or not state.is_file() or read_json(state).get("status") != "building":
            raise ValueError("Output exists; use --check, or --resume for an unfinished matching build only")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit").mkdir(exist_ok=True)
    signature = digest(protocol)
    if state.exists() and read_json(state)["protocol_sha256"] != signature:
        raise ValueError("Cannot resume a different protocol")
    write_json(state, {"protocol_sha256": signature, "status": "building"})
    repo = root / protocol["source"]["local_repository"]
    tree = source_tree(repo, protocol["source"]["snapshot_commit"])
    status_before = git(repo, "status", "--porcelain", "--untracked-files=no")
    log(output, "source_verified", commit=protocol["source"]["snapshot_commit"], git_status=status_before)
    cache = output / "audit/inventory_cache.jsonl"
    if cache.exists() and resume:
        rows = read_jsonl(cache)
        def verify_cached_source(row):
            if file_hash(root / row["original_path"]) != row["sha256"]:
                raise ValueError("Source image changed since interrupted inventory")
        with ThreadPoolExecutor(max_workers=12) as pool:
            for _ in pool.map(verify_cached_source, rows):
                pass
        log(output, "resumed_inventory", images=len(rows))
    else:
        rows = inventory_paths(root, protocol)
        log(output, "inventory_started", images=len(rows))
        with ThreadPoolExecutor(max_workers=12) as pool:
            rows = list(pool.map(partial(inspect_image, root, tree), rows))
        write_jsonl(cache, rows)
        log(output, "inventory_completed", images=len(rows), readable=sum(r["image_status"] == "readable" for r in rows))
    rows.sort(key=lambda row: row["sample_id"])
    csv_audit = audit_group_csv(root, repo, output, tree, rows)
    log(output, "group_csv_verified_and_recreated", committed=csv_audit["committed_counts"], regenerated=csv_audit["regenerated_counts"])
    groups = exact_components(rows)
    assign_proposed_splits(rows, protocol["split_policy"]["seed"])
    candidates = phash_candidates(rows, protocol["split_policy"]["phash_distance"])
    isolate_candidates(rows, candidates)
    write_jsonl(output / "audit/phash_candidates.jsonl", candidates)
    write_jsonl(output / "audit/exact_duplicate_groups.jsonl", [dict(group_id=group[0]["duplicate_group"],
        samples=[{key: r.get(key) for key in ("sample_id", "source", "source_path", "label", "role", "condition", "sha256", "pixel_sha256")} for r in group])
        for group in groups.values() if len(group) > 1])
    log(output, "split_proposed_and_duplicates_audited", splits=dict(Counter(r["split"] for r in rows)),
        phash_candidates=len(candidates), confirmed_duplicate_groups=sum(len(g) > 1 for g in groups.values()))
    reused = reuse_annotations(root, protocol, rows)
    add_csv_proposals(repo, tree, rows)
    propose_capture_boxes(root, protocol, rows)
    flag_annotations(rows)
    log(output, "annotation_prepared", reused=reused, states=dict(Counter(r["annotation_state"] for r in rows)))
    with ThreadPoolExecutor(max_workers=12) as pool:
        for _ in pool.map(partial(export_one, root, output), rows):
            pass
    write_lists_and_configs(output, rows)
    log(output, "dataset_exported", images=sum(bool(r.get("image")) for r in rows), labels=sum(bool(r.get("yolo_label")) for r in rows))
    loader = previews_and_loader(output, protocol, rows)
    log(output, "loader_and_previews_verified", train_batch=loader["baseline"]["batch_shape"], val_batch=loader["val"]["batch_shape"])
    summary = summarize(rows)
    fingerprint = manifest_fingerprint(rows)
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "protocol.json", protocol)
    write_json(output / "manifest.json", {"schema_version": 1, "kind": "asl29_image_level_manifest", "rows_file": "manifest.jsonl",
                                          "classes": CLASSES_29, "fingerprint": fingerprint, "protocol_sha256": signature,
                                          "split_status": "proposed", "split_frozen": False,
                                          "source": protocol["source"], "csv_hashes": csv_audit["csv_hashes"],
                                          "prior_annotation_manifest_sha256": file_hash(root / protocol["source"]["reviewed_snapshot"] / "manifest.json")})
    fields = ["sample_id", "source", "source_path", "original_path", "normalized_path", "sha256", "pixel_sha256", "width", "height", "label", "member", "condition", "session_id", "burst_id", "role", "split", "split_status", "duplicate_group", "split_group", "image_status", "annotation_state", "annotation_reason", "capture_type", "exclusions", "image", "yolo_label", "exported_for_loader"]
    numbers = Counter()
    for row in rows:
        if row["member"]:
            key = row["member"], row["label"]
            numbers[key] += 1
            row["normalized_path"] = f"{row['member']}_dataset/{row['label']}/{row['label']}_{row['member']}_{numbers[key]:02d}" + (".jpg" if row.get("format") == "JPEG" else ".png")
    # normalized_path is an organizational map, not a claim that originals were renamed.
    write_jsonl(output / "manifest.jsonl", rows)
    write_csv(output / "manifest.csv", rows, fields)
    for split in (*SPLITS, "external_test", "excluded"):
        write_csv(output / "splits" / f"{split}.csv", [r for r in rows if r["split"] == split], fields)
    write_csv(output / "audit/rename_mapping.csv", [r for r in rows if r["member"]],
              ["sample_id", "original_path", "normalized_path", "sha256", "member", "condition", "role"])
    write_csv(output / "audit/review_queue.csv", [r for r in rows if r["annotation_state"] != "reviewed" or r["exclusions"]], fields)
    write_csv(output / "audit/distribution.csv", [dict(split=s, source=src, member=m, condition=c, label=l, images=n)
              for (s, src, m, c, l), n in sorted(Counter((r["split"], r["source"], r["member"] or "unknown", r["condition"] or "unknown", r["label"]) for r in rows).items())],
              ["split", "source", "member", "condition", "label", "images"])
    technical = verify(output)
    if git(repo, "status", "--porcelain", "--untracked-files=no") != status_before:
        raise ValueError("Reference repository changed during preprocessing")
    missing = [label for label in CLASSES_29 if not summary["splits"]["train"]["class_positive_boxes"].get(label)]
    blockers = ["split_status=proposed; group must approve/freeze membership", "human ground-truth confirmation required; automated reviews remain automated",
                "pending/rejected annotations and ambiguous nothing types must be resolved", "holdout ground-truth coverage is incomplete; annotated subset is not a complete benchmark"]
    if candidates:
        blockers.append("pHash candidates require identity review; cross-split development candidates isolated from loader lists")
    if missing:
        blockers.append("no exported positive train boxes for classes: " + ", ".join(missing))
    report = dict(**summary, **technical, blockers=blockers, split_status="proposed", split_frozen=False,
                  group_manifest_audit=csv_audit, source_repository_unchanged=True,
                  versions={p: importlib.metadata.version(p) for p in ("ultralytics", "mediapipe", "torch", "Pillow", "imagehash", "numpy")},
                  prior_model_independence="not established for previously trained checkpoints; November split is a new proposal")
    write_json(output / "audit/report.json", report)
    write_json(state, {"protocol_sha256": signature, "status": "complete", "fingerprint": fingerprint, "training_ready": False})
    log(output, "preprocessing_complete", fingerprint=fingerprint, training_ready=False, blockers=blockers)
    return {"output": str(output), **technical, "splits": summary["splits"], "blockers": blockers}
