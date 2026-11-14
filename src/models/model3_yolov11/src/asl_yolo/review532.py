"""Hybrid sampling QC for the image-level November draft; never impersonates a human.

The immutable base retains every holdout image. Machine results are evidence, not
human approvals. A release requires representative human checks, per-image review
of critical nothing cases, and a frozen, leakage-checked run list.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import ast
import importlib.metadata
import importlib.util
from html import escape
import json
import os
from pathlib import Path
import random
import shutil
from threading import local
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps
import yaml

from .core import digest, file_hash, read_json, write_json
from .dataset29 import load_review
from .preprocess532 import (
    augmentation_config, draw_tile, manifest_fingerprint, map_padded_points,
    make_loader, read_jsonl, save_sheet, write_csv,
)
from .protocol import CLASSES_29, validate_annotation
from .vision import HandTracker, landmarks_box, xyxy_to_yolo

FIT = {"train", "val"}
METHOD = "november_hybrid_sampling_qc_v1"
LOCAL = local()


def utc():
    return datetime.now(timezone.utc).isoformat()


def event(output, step, **data):
    with (output / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": utc(), "step": step, **data}, ensure_ascii=False) + "\n")


def load_base(dataset):
    manifest = read_json(dataset / "manifest.json")
    rows = read_jsonl(dataset / manifest.get("rows_file", "manifest.jsonl"))
    if tuple(manifest["classes"]) != CLASSES_29 or manifest_fingerprint(rows) != manifest["fingerprint"]:
        raise ValueError("Base manifest/class mapping changed")
    return manifest, rows


def eligible(row):
    return row["split"] in FIT and row["role"] == "development" and row["image_status"] == "readable" and not row["exclusions"]


def stratum(row):
    return "/".join((row["split"], row["source"], row["label"]))


def load_helpers(root):
    # Reuse the prior project's fixed geometry/agreement rules, rather than tuning on test.
    path = root / "scripts/auto_review_asl29.py"
    spec = importlib.util.spec_from_file_location("asl29_existing_qa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine_code_fingerprint(source):
    tree = ast.parse(source)
    return digest([ast.dump(node, include_attributes=False) for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in {"candidate_box", "inspect_sample"}])


def assert_compatible_round(meta, signature, output):
    saved = dict(meta["signature"])
    old_source = output / "source_snapshot/review532.py"
    if "machine_code_sha256" not in saved:
        if not old_source.is_file() or file_hash(old_source) != saved["pipeline_sha256"]:
            raise ValueError("Missing exact source snapshot for this review round")
        saved["machine_code_sha256"] = machine_code_fingerprint(old_source.read_text(encoding="utf-8"))
    # UI/assessment changes may evolve without rerunning MediaPipe. The exact
    # extraction/QA functions, helper, model, dataset and versions stay locked.
    current = dict(signature)
    saved.pop("pipeline_sha256", None)
    current.pop("pipeline_sha256", None)
    if saved != current:
        raise ValueError("Review round inputs or machine QA changed; use a new --output")


def candidate_box(row, tracker, rgb):
    import cv2
    attempts = []
    if len(row["boxes"]) == 1:
        return row["boxes"], attempts
    if "multiple_hands" in row["flags"]:
        return [], [{"reason": "previous_multiple_hand_flag_requires_visual_review"}]
    for mode, border in (("original", 0), ("mirror", 0), ("black", 50), ("replicate", 50), ("replicate", 80)):
        frame = rgb[:, ::-1] if mode == "mirror" else rgb
        if border:
            frame = cv2.copyMakeBorder(frame, border, border, border, border,
                                       cv2.BORDER_REPLICATE if mode == "replicate" else cv2.BORDER_CONSTANT)
        hands = tracker.detect(frame)
        attempts.append({"mode": mode, "border_px": border, "hands": len(hands)})
        if len(hands) > 1:
            return [], attempts
        if len(hands) != 1:
            continue
        points = map_padded_points(hands[0], row["width"], row["height"], border)
        if mode == "mirror":
            points[:, 0] = 1 - points[:, 0]
        try:
            box = xyxy_to_yolo(landmarks_box(points, row["width"], row["height"], .15), row["width"], row["height"])
            validate_annotation(row["label"], row["capture_type"], [box])
            return [box], attempts
        except ValueError:
            continue
    return [], attempts


def inspect_sample(dataset, row, helpers):
    if not eligible(row):
        raise ValueError("Automated QC may only inspect eligible development images")
    path = dataset / row["image"]
    if file_hash(path) != row["output_sha256"]:
        raise ValueError(f"Changed image: {row['sample_id']}")
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    result = dict(state="needs_review", boxes=row["boxes"], sha256=row["sha256"],
                  output_sha256=row["output_sha256"], reviewer="MediaPipe automated QC",
                  review_kind="automated", method=METHOD, reviewed_at=utc(), stratum=stratum(row),
                  note="Automated evidence only; source class is not independently confirmed by landmarks.", evidence={})
    if row["capture_type"] in {"other_hand", "unresolved"}:
        result["note"] = "Human confirmation of nothing subtype and out-of-vocabulary pose is required; never infer it from detector failure."
        result["evidence"]["critical_semantic_review"] = True
        return result
    if row["capture_type"] == "no_hand":
        counts = [len(LOCAL.tracker.detect(frame)) for frame in (rgb, rgb[:, ::-1])]
        result["evidence"]["hand_counts"] = counts
        result["boxes"] = []
        if not any(counts):
            result.update(state="machine_pass", note="Declared background source corroborated by two detector passes; sampling QC still required. MP failure alone is not a background label.")
        else:
            result["note"] = "Declared background flagged for a possible hand; requires visual review."
        return result
    boxes, attempts = candidate_box(row, LOCAL.tracker, rgb)
    result["evidence"]["proposal_attempts"] = attempts
    result["boxes"] = boxes
    if not boxes:
        result["note"] = "No reliable single-hand proposal; keep the image pending in its original split."
        return result
    proposed = {**row, "boxes": boxes}
    if (row.get("annotation_state") == "reviewed" and row.get("review_history")
            and row["review_history"][-1].get("method") == "asl29_screened_crosscheck_v1"):
        # The prior release already expanded these boxes 20% per side. Invert that
        # known expansion before reusing its comparison/re-expansion procedure.
        x, y, width, height = boxes[0]
        proposed["boxes"] = [[x, y, width / 1.4, height / 1.4]]
        result["evidence"]["prior_release_padding_inverted_for_comparison"] = .20
    crosscheck = helpers.screen_sign(proposed, helpers.crosscheck_sign(proposed, LOCAL.tracker, rgb))
    result["evidence"]["crosscheck"] = crosscheck
    if crosscheck["state"] == "approved":
        result.update(state="machine_pass", boxes=crosscheck["boxes"],
                      note="Two views corroborate geometry under fixed prior QA rules. ASL folder label retained; representative human QC is required.")
    else:
        result["note"] = crosscheck["note"]
    return result


def sampling_plan(rows, evidence, per_stratum=3, seed=42):
    groups, selected = defaultdict(list), set()
    for row in rows:
        if eligible(row) and evidence[row["sample_id"]]["state"] == "machine_pass":
            groups[stratum(row)].append(row["sample_id"])
    strata = {}
    for key, ids in sorted(groups.items()):
        ids.sort()
        random.Random(digest([seed, key])).shuffle(ids)
        chosen = ids[:per_stratum]
        selected.update(chosen)
        strata[key] = {"population": len(ids), "sample_ids": chosen, "sampling_status": "pending_human_review"}
    # Cover every eligible member/condition as well as the class/source/split strata.
    for dimension in ("member", "condition"):
        groups_by_value = defaultdict(list)
        for row in rows:
            if eligible(row) and evidence[row["sample_id"]]["state"] == "machine_pass" and row.get(dimension):
                groups_by_value[row[dimension]].append(row["sample_id"])
        for ids in groups_by_value.values():
            if not selected.intersection(ids):
                selected.add(min(ids))
    by_id = {r["sample_id"]: r for r in rows}
    for key in selected:
        group = strata[stratum(by_id[key])]
        if key not in group["sample_ids"]:
            group["sample_ids"].append(key)
    critical = [r["sample_id"] for r in rows if eligible(r) and r["class_id"] == 27 and r["capture_type"] != "no_hand"]
    return dict(seed=seed, per_stratum=per_stratum, strata=strata, representative_ids=sorted(selected),
                mandatory_semantic_ids=critical, human_decisions_required=True,
                annotation_policy="sampling_QC_for_machine_pass; individual_review_for_critical_or_rescued_samples",
                failed_sample_policy="block its entire stratum until reviewed/reworked; do not simply remove a failed sampled image")


def validate_human_decision(row, decision):
    if decision.get("sha256") != row["sha256"]:
        raise ValueError("Stale human decision")
    if decision.get("review_kind") != "human" or not decision.get("reviewer", "").strip() or not decision.get("reviewed_at"):
        raise ValueError("Explicit human reviewer, timestamp and review_kind are required")
    if decision.get("state") not in {"approved", "rejected"}:
        raise ValueError("Unknown human decision state")
    if not decision.get("note", "").strip():
        raise ValueError("Human decisions must record label/coverage verification or rejection reason")
    if decision["state"] == "approved":
        capture_type = decision.get("capture_type", row["capture_type"])
        if any("multiple_hands" in flag for flag in row["flags"]) and not decision.get("single_hand_confirmed"):
            raise ValueError("Multiple-hand cases require per-object labels or explicit single-hand metadata correction")
        validate_annotation(row["label"], capture_type, decision["boxes"])


def assess_release(rows, evidence, plan, human):
    by_id = {r["sample_id"]: r for r in rows}
    for key, decision in human.items():
        if key not in by_id or not eligible(by_id[key]):
            raise ValueError("Human decision cannot change holdout membership or release a quarantined image")
        validate_human_decision(by_id[key], decision)
    statuses, selected = {}, {}
    for key, group in plan["strata"].items():
        decisions = [human.get(i) for i in group["sample_ids"]]
        checked_ids = [i for i in human if stratum(by_id[i]) == key and evidence[i]["state"] == "machine_pass"]
        checked = [(i, human[i]) for i in checked_ids]
        changed = any(d and (d["state"] == "rejected" or d.get("proposal_changed") or
                            len(d["boxes"]) != len(evidence[i]["boxes"]) or
                            any(abs(a - b) > 1e-6 for x, y in zip(d["boxes"], evidence[i]["boxes"]) for a, b in zip(x, y)))
                      for i, d in checked)
        statuses[key] = ("failed_sample" if changed
                         else "passed" if all(d and d["state"] == "approved" for d in decisions)
                         else "pending")
    for row in rows:
        key = row["sample_id"]
        if not eligible(row):
            continue
        decision = human.get(key)
        if decision:
            if decision["state"] == "approved" and statuses.get(stratum(row)) != "failed_sample":
                selected[key] = {"boxes": decision["boxes"], "acceptance": "individual_human_review",
                                 "capture_type": decision.get("capture_type", row["capture_type"])}
            continue
        if (row["capture_type"] not in {"other_hand", "unresolved"} and evidence[key]["state"] == "machine_pass"
                and statuses.get(stratum(row)) == "passed"):
            selected[key] = {"boxes": evidence[key]["boxes"], "acceptance": "automated_geometry_with_human_sampling_QC",
                             "capture_type": row["capture_type"]}
    pending_samples = [i for i in plan["representative_ids"] if i not in human]
    # Semantic samples that remain pending are excluded, not silently relabelled.
    positive_train = Counter(r["label"] for r in rows if r["split"] == "train" and r["sample_id"] in selected
                             for _ in selected[r["sample_id"]]["boxes"])
    missing = [c for c in CLASSES_29 if not positive_train[c]]
    blockers = []
    if pending_samples:
        blockers.append(f"Representative human QC still missing for {len(pending_samples)} images")
    if "failed_sample" in statuses.values():
        blockers.append("A failed representative sample blocks its entire stratum")
    if missing:
        blockers.append("Missing accepted positive train boxes: " + ", ".join(missing))
    if not any(r["split"] == "val" and r["sample_id"] in selected for r in rows):
        blockers.append("No accepted validation images")
    return dict(stratum_status=statuses, selected=selected, positive_train_boxes=dict(positive_train),
                missing_train_classes=missing, pending_representative_ids=pending_samples,
                reviewed_human_images=len(human), blockers=blockers, training_ready=False,
                note="A separate explicit freeze and final integrity/leakage audit are required even after QC passes.")


def freeze_release(dataset, output, release):
    """Materialize an accepted run subset, preserving the base and complete holdout."""
    manifest, rows = load_base(dataset)
    evidence = load_review(output / "evidence.jsonl")["annotations"]
    plan = read_json(output / "sampling_plan.json")
    meta = read_json(output / "round.json")
    if meta["signature"]["base_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Review round belongs to another base dataset")
    by_id = {r["sample_id"]: r for r in rows if eligible(r)}
    if set(evidence) != set(by_id):
        raise ValueError("Automated QC must cover every eligible development image")
    for key, decision in evidence.items():
        if (decision.get("method") != METHOD or decision["sha256"] != by_id[key]["sha256"]
                or decision.get("output_sha256") != by_id[key]["output_sha256"]):
            raise ValueError("Stale automated QC evidence")
    if plan != sampling_plan(rows, evidence):
        raise ValueError("Representative sampling plan changed")
    human_path = output / "human_decisions.jsonl"
    human = load_review(human_path).get("annotations", {}) if human_path.exists() else {}
    assessment = assess_release(rows, evidence, plan, human)
    if assessment["blockers"]:
        raise ValueError("Freeze blocked: " + "; ".join(assessment["blockers"]))
    return materialize_release(dataset, output, release, manifest, rows, assessment["selected"], {
        "human_decisions_sha256": file_hash(human_path), "evidence_sha256": file_hash(output / "evidence.jsonl"),
        "sampling_plan_sha256": file_hash(output / "sampling_plan.json"),
        "review_policy": "automated_geometry_with_human_sampling_QC_and_individual_critical_reviews",
        "review_kind": "hybrid", "annotation_quality": "human_sampling_QC_with_individual_critical_reviews",
    })


def materialize_release(dataset, output, release, manifest, rows, selected, review_proof):
    """Shared checked exporter for explicitly documented human or automated policies."""
    if release.exists():
        raise ValueError("Release already exists; choose a new version")
    positives = Counter(r["label"] for r in rows if r["split"] == "train" and r["sample_id"] in selected
                        for _ in selected[r["sample_id"]]["boxes"])
    if any(not positives[c] for c in CLASSES_29) or not any(r["split"] == "val" and r["sample_id"] in selected for r in rows):
        raise ValueError("Release requires positive train boxes for all 29 classes and a nonempty validation set")
    leakage = duplicate_check(dataset, rows, selected)
    release.mkdir(parents=True)
    write_json(release / "build_state.json", {"status": "building", "training_ready": False})
    lists = {"train": [], "val": []}
    for split in lists:
        (release / "draft/images" / split).mkdir(parents=True)
        (release / "draft/labels" / split).mkdir(parents=True)

    def emit(row):
        key = row["sample_id"]
        annotation = selected[key]
        validate_annotation(row["label"], annotation["capture_type"], annotation["boxes"])
        source = dataset / row["image"]
        if file_hash(source) != row["output_sha256"]:
            raise ValueError("A reviewed source image changed")
        image = Path("draft/images", row["split"], key + source.suffix)
        label = Path("draft/labels", row["split"], key + ".txt")
        shutil.copyfile(source, release / image)
        content = "".join(str(row["class_id"]) + " " + " ".join(f"{v:.10f}" for v in box) + "\n" for box in annotation["boxes"])
        (release / label).write_text(content, encoding="ascii")
        if file_hash(release / image) != row["output_sha256"]:
            raise ValueError("Release image copy differs")
        return dict(sample_id=key, split=row["split"], label=row["label"], class_id=row["class_id"],
                            capture_type=annotation["capture_type"], source_capture_type=row["capture_type"],
                            image=image.as_posix(), image_sha256=row["output_sha256"],
                            label_file=label.as_posix(), label_sha256=file_hash(release / label), boxes=annotation["boxes"],
                            acceptance=annotation["acceptance"], source_image=row["image"], source_sha256=row["sha256"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        emitted = list(pool.map(emit, [r for r in rows if r["sample_id"] in selected]))
    for row in emitted:
        lists[row["split"]].append("./" + Path(row["image"]).relative_to("draft").as_posix())
    for split, paths in lists.items():
        (release / "draft" / (split + ".txt")).write_text("\n".join(paths) + "\n", encoding="utf-8")
    data = {"path": (release / "draft").as_posix(), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(CLASSES_29))}
    (release / "draft/data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    protocol = read_json(dataset / "protocol.json")
    loaders = {}
    for split in ("train", "val"):
        loader = make_loader(release, split, augmentation_config(protocol), split == "train")
        batch = loader.collate_fn([loader[i] for i in range(min(4, len(loader)))])
        if len(loader) != len(lists[split]) or batch["img"].shape[1:] != (3, 320, 320):
            raise ValueError("Release loader membership or shape mismatch")
        loaders[split] = {"images": len(loader), "batch_shape": list(batch["img"].shape)}
    for mirror in (False, True):
        config = augmentation_config(protocol, mirror)
        (release / ("augmentation_mirror.yaml" if mirror else "augmentation_baseline.yaml")).write_text(yaml.safe_dump(config), encoding="utf-8")
    def verify_emitted(row):
        if file_hash(release / row["image"]) != row["image_sha256"] or file_hash(release / row["label_file"]) != row["label_sha256"]:
            raise ValueError("Release bytes changed during loader checks")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify_emitted, emitted):
            pass
    frozen = dict(kind="asl29_qc_run_release", classes=CLASSES_29, base_manifest_fingerprint=manifest["fingerprint"],
                  source_split_status="proposed", team_split_approved=False, run_split_frozen=True,
                  freeze_scope="accepted train/val subset only; source split and complete holdout unchanged",
                  fingerprint=digest(emitted), samples=emitted, counts={s: len(p) for s, p in lists.items()},
                  **review_proof,
                  training_ready=True, evaluation_ready=False, loaders=loaders, leakage=leakage,
                  holdout_reference={"base_dataset": str(dataset), "complete_manifest": "manifest.jsonl",
                                     "group_test": "holdout/group_test_all.txt", "external_test": "holdout/external_test_canonical_all.txt",
                                     "ground_truth_coverage_incomplete": True},
                  warnings=["Validation has no positive nothing/other_hand from an independent session", "This is a filtered run subset; do not claim full group-data training or complete test evaluation"])
    write_json(release / "release.json", frozen)
    write_json(release / "training_plan.json", {"model": "yolo11n.pt", "imgsz": 320, "batch": 8, "seed": 42,
                                               "freeze": 0, "smoke_epochs": 3, "baseline_epochs": 50, "require_cuda": True,
                                               "allow_training": False, "allow_final_test": False,
                                               "augmentation": "augmentation_baseline.yaml"})
    write_json(release / "build_state.json", {"status": "complete", "fingerprint": frozen["fingerprint"], "training_ready": True})
    return {k: frozen[k] for k in ("training_ready", "evaluation_ready", "fingerprint", "counts")}


def duplicate_check(dataset, rows, ids):
    ids = set(ids)
    by_id = {r["sample_id"]: r for r in rows}
    violations = []
    for pair in read_jsonl(dataset / "audit/phash_candidates.jsonl"):
        if pair["cross_split"] and (pair["a"] in ids or pair["b"] in ids):
            violations.append(pair)
    groups = defaultdict(set)
    for row in rows:
        if row["sample_id"] in ids:
            if row["role"] != "development" or row["split"] not in FIT or row["exclusions"]:
                raise ValueError("Unsafe sample in candidate run")
            for field in ("sha256", "pixel_sha256", "split_group"):
                groups[(field, row[field])].add(row["split"])
    if violations or any(len(v) > 1 for v in groups.values()):
        raise ValueError("Candidate run has a hash/session/pHash cross-split conflict")
    return dict(candidate_images=len(ids), confirmed_cross_split_conflicts=0, unresolved_cross_split_candidates_in_run=0,
                policy="All base quarantine decisions retained. No pHash pair was automatically declared a duplicate.")


def make_packet(root, dataset, output, rows, evidence):
    plan = sampling_plan(rows, evidence)
    if (output / "sampling_plan.json").exists() and read_json(output / "sampling_plan.json") != plan:
        raise ValueError("Sampling plan changed; use a new review round instead of replacing reviewed samples")
    write_json(output / "sampling_plan.json", plan)
    by_id = {r["sample_id"]: r for r in rows}
    for name, ids in (("representative", plan["representative_ids"]), ("critical_nothing", plan["mandatory_semantic_ids"]),
                      ("exceptions", [r["sample_id"] for r in rows if eligible(r) and evidence[r["sample_id"]]["state"] != "machine_pass"] )):
        (output / f"{name}_ids.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
        write_csv(output / f"{name}_queue.csv", [dict(**{k: by_id[i].get(k) for k in ("sample_id", "split", "source", "label", "member", "condition", "capture_type", "sha256", "image", "flags")},
                  state=evidence[i]["state"], reason=evidence[i]["note"]) for i in ids],
                  ["sample_id", "split", "source", "label", "member", "condition", "capture_type", "sha256", "image", "flags", "state", "reason"])
    preview = output / "previews"
    preview.mkdir(exist_ok=True)
    pages = []
    ids = list(dict.fromkeys(plan["mandatory_semantic_ids"] + plan["representative_ids"]))
    for start in range(0, len(ids), 20):
        tiles = []
        for key in ids[start:start + 20]:
            row = by_id[key]
            with Image.open(dataset / row["image"]) as image:
                tiles.append(draw_tile(image, evidence[key]["boxes"], f"{row['label']} {row['split']} {key[:8]}"))
        path = preview / f"qc_{start // 20 + 1:03d}.jpg"
        save_sheet(path, tiles)
        pages.append({"page": path.relative_to(output).as_posix(), "sample_ids": ids[start:start + 20]})
    write_json(output / "preview_index.json", pages)
    page_cards = []
    for page in pages:
        links = []
        for key in page["sample_ids"]:
            row = by_id[key]
            relative = Path(os.path.relpath(dataset / row["image"], output)).as_posix()
            links.append(f'<a href="{quote(relative)}">{escape(row["label"])} · {escape(row["split"])} · {key[:8]}</a>')
        page_cards.append(f'<section><h2>{escape(page["page"])}</h2><img src="{page["page"]}" alt="QC contact sheet"><p>{" | ".join(links)}</p></section>')
    (output / "index.html").write_text('''<!doctype html><html lang="vi"><meta charset="utf-8"><title>ASL29 — QC trước train</title>
<style>body{margin:32px auto;max-width:1250px;padding:0 20px;font:16px/1.6 system-ui;background:#101820;color:#edf2f4}a{color:#86d6ed}section{margin:36px 0;padding:20px;background:#1b2934;border-radius:12px}img{max-width:100%;height:auto}h1{line-height:1.2}.warning{color:#ffd189}</style>
<h1>ASL29 — Kiểm tra theo mẫu và ca bất thường</h1><p class="warning">Proposal của máy chưa phải ground truth đã được người duyệt. Chưa train.</p>
<p>Xem toàn ảnh qua các liên kết dưới bảng; đối chiếu nhãn, đầu ngón, cổ tay, tay phụ. Quyết định chỉ được lưu khi dùng editor; xem README.md và scripts/start_group_532_review.ps1.</p>'''
        + "\n".join(page_cards) + "</html>", encoding="utf-8")
    candidate_ids = [r["sample_id"] for r in rows if eligible(r) and evidence[r["sample_id"]]["state"] == "machine_pass"]
    leakage = duplicate_check(dataset, rows, candidate_ids)
    write_json(output / "leakage_audit.json", leakage)
    human_path = output / "human_decisions.jsonl"
    human = load_review(human_path).get("annotations", {}) if human_path.exists() else {}
    assessment = assess_release(rows, evidence, plan, human)
    write_json(output / "release_assessment.json", assessment)
    stats = dict(total_base_images=len(rows), eligible_development_images=len(evidence),
                 machine_status=dict(Counter(d["state"] for d in evidence.values())),
                 by_split={s: dict(Counter(evidence[r["sample_id"]]["state"] for r in rows if eligible(r) and r["split"] == s)) for s in sorted(FIT)},
                 new_proposals_on_previously_boxless=sum(not r["boxes"] and bool(evidence[r["sample_id"]]["boxes"]) for r in rows if eligible(r)),
                 representative_images=len(plan["representative_ids"]), critical_nothing_images=len(plan["mandatory_semantic_ids"]),
                 qc_strata=len(plan["strata"]), source_split_unchanged=True, training_ready=False, leakage=leakage)
    write_json(output / "summary.json", stats)
    distribution = Counter((r["split"], r["source"], r["member"] or "unknown", r["condition"] or "unknown", r["label"],
                            evidence[r["sample_id"]]["state"]) for r in rows if eligible(r))
    write_csv(output / "quality_distribution.csv", [dict(split=s, source=src, member=m, condition=c, label=l, machine_state=state, images=n)
              for (s, src, m, c, l, state), n in sorted(distribution.items())],
              ["split", "source", "member", "condition", "label", "machine_state", "images"])
    text = f"""# Bước tiếp theo: QC theo mẫu và duyệt ca bất thường

Nguồn kế hoạch: `reports/asl_alphabet_rerun_20261102/PLAN.md`, Giai đoạn B/C.
Đây là kế hoạch đang lưu trong dự án; chưa có bằng chứng xác nhận riêng của trưởng nhóm.
Giữ nguyên manifest tháng 11, split đề xuất 5/3/2, đầy đủ holdout và hai điều kiện Bao.
Chưa train hoặc đánh giá model. `training_ready=false`.

Đã kiểm tra tự động {len(evidence):,} ảnh development đủ điều kiện (không bị cách ly).
Trạng thái: `{json.dumps(stats['machine_status'])}`. Có {stats['new_proposals_on_previously_boxless']:,} ảnh trước đó chưa có box được bổ sung proposal.
Mọi kết quả máy được ghi `review_kind=automated`; `machine_pass` chỉ là bằng chứng hình học, không phải người duyệt hoặc xác nhận đúng ký hiệu ASL.
Quy tắc kiểm tra tái sử dụng script QA V3: IoU ≥0.88, box tối thiểu 12% mỗi chiều, mở rộng 20%, biên tối thiểu 0.008.
Các quy tắc đã chọn từ train ở thí nghiệm trước; không dùng test để chỉnh tham số. Hai lượt cùng MediaPipe có thể cùng sai.

## Thứ tự duyệt

1. Duyệt {stats['critical_nothing_images']} ảnh trong `critical_nothing_ids.txt`: xác nhận no_hand/other_hand, tư thế ngoài 28 ký hiệu, box đủ bàn tay. Không lấy lỗi MediaPipe làm nhãn nothing.
2. Duyệt {stats['representative_images']} ảnh trong `representative_ids.txt`, tối đa 3 ảnh mỗi nhóm `(split, source, label)`, bổ sung bao phủ member/condition. Đây là quy trình sampling QC, không phải duyệt tay toàn bộ dataset.
3. Ca không đạt kiểm tra máy nằm ở `exceptions_queue.csv`. Có thể sửa từng ảnh để bổ sung vào run; ảnh chưa sửa tiếp tục pending, không đổi split hoặc giả nhãn. Một mẫu đại diện bị từ chối chặn cả nhóm tương ứng để kiểm tra lại, không chỉ bỏ mẫu xấu rồi cho nhóm qua.
4. Sau review, chạy `--assess`. Chỉ khi các điều kiện QC đạt mới được khóa danh sách run riêng; không tuyên bố nhóm đã chốt split chính thức. Holdout thiếu ground truth vẫn chặn đánh giá test, không tự chọn subset dễ làm benchmark.

Mức 3 ảnh/nhóm là đề xuất QC khởi đầu của pipeline, chưa phải số lượng trưởng nhóm xác nhận và không bảo đảm một tỷ lệ sai nhãn thống kê. Danh sách mẫu được khóa theo seed 42; không thay mẫu đã bị phát hiện lỗi để làm QC đạt.

## Windows

```powershell
.\\.venv\\Scripts\\python.exe scripts/review_asl_group_532.py --prepare --workers 4
.\\.venv\\Scripts\\python.exe scripts/review_asl29.py --dataset data/processed/asl_yolo11n_group_532_v1 --evidence data/review/asl_yolo11n_group_532_v1/round_01/evidence.jsonl --sample-list data/review/asl_yolo11n_group_532_v1/round_01/critical_nothing_ids.txt --log data/review/asl_yolo11n_group_532_v1/round_01/human_decisions.jsonl --include-uncertain --reviewer Bao
.\\.venv\\Scripts\\python.exe scripts/review_asl29.py --dataset data/processed/asl_yolo11n_group_532_v1 --evidence data/review/asl_yolo11n_group_532_v1/round_01/evidence.jsonl --sample-list data/review/asl_yolo11n_group_532_v1/round_01/representative_ids.txt --log data/review/asl_yolo11n_group_532_v1/round_01/human_decisions.jsonl --reviewer Bao
.\\.venv\\Scripts\\python.exe scripts/review_asl_group_532.py --assess
# Chỉ sau khi QC thực tế đạt; lệnh tự chặn nếu thiếu quyết định người duyệt:
.\\.venv\\Scripts\\python.exe scripts/review_asl_group_532.py --freeze --release data/processed/asl_yolo11n_group_532_qc_v2
```

Editor: kéo chuột thay box; A duyệt, R từ chối, C xóa, S bỏ qua, Q thoát. Với nhãn nothing, N xác nhận no_hand hoặc O xác nhận other_hand (quyết định người duyệt, không suy ra từ máy). Phím 1 chỉ dùng khi đã nhìn toàn ảnh và xác nhận cờ nhiều tay là false positive, thực tế chỉ một tay. Luôn ghi lý do kiểm tra nhãn và độ bao phủ tay. Có `--list` để đếm queue, `--redo` để append quyết định sửa lại; lịch sử log vẫn giữ nguyên.
Không mở webcam. Ảnh thực sự nhiều tay phải để chờ công cụ gán class riêng cho từng object.
Mở `index.html` để xem contact sheet và bấm từng sample để xem toàn ảnh. Khởi động nhanh từ project root:

```powershell
& .\\scripts\\start_group_532_review.ps1
& .\\scripts\\start_group_532_review.ps1 -Queue representative
```

## Trùng ảnh và phạm vi train

{leakage['candidate_images']:,} ứng viên máy đạt QC không có exact/session/pHash candidate cắt ngang các split đang dùng.
Giữ cách ly toàn bộ ứng viên pHash từ bản gốc; không tự kết luận chúng là cùng ảnh. Để mở rộng dữ liệu từ phần cách ly, cần review riêng.
Số ảnh nguồn vẫn giữ nguyên; subset có thể dùng cho run sau QC nhỏ hơn 44.261/26.520. Cần công bố đúng phạm vi, không nói đã train toàn bộ nguồn nhóm.
Validation chưa có positive class 27 từ session độc lập; cần bổ sung development mới trước khi chọn ngưỡng/đánh giá riêng lớp đó. Không lấy ảnh Bao để bù.

Bước train tiếp theo theo kế hoạch: smoke 1–3 epoch trên GPU Colab, sau đó baseline YOLO11n 50 epoch, imgsz=320, batch=8, seed=42, freeze=0, augmentation light đã khóa. Chưa kích hoạt trong tác vụ QC này.
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    report_dir = root / "reports/asl_yolo11n_group_532_review_01"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "REVIEW_REPORT.md").write_text(text, encoding="utf-8")
    steps = ["# Các bước quan trọng — QC trước train", "", f"Base fingerprint: `{read_json(dataset / 'manifest.json')['fingerprint']}`.",
             "", "Giờ log dùng UTC; giờ Việt Nam = UTC+7. Mọi kết quả máy là automated; chưa có quyết định người duyệt thì không khóa run.", ""]
    for record in read_jsonl(output / "steps.jsonl"):
        if record["step"] == "qc_progress":
            continue
        steps += [f"## {record['step']}", "", record["at_utc"], "", "```json",
                  json.dumps({k: v for k, v in record.items() if k not in {"step", "at_utc"}}, indent=2), "```", ""]
    steps += ["## Kết quả hiện tại", "", "```json", json.dumps(stats, indent=2), "```", "",
              "QC đại diện và các ca nothing cần người duyệt thực tế; không có chữ ký thay người dùng. Các ảnh khó vẫn nằm trong manifest gốc và split gốc."]
    (report_dir / "IMPORTANT_STEPS.md").write_text("\n".join(steps), encoding="utf-8")
    checksums = {}
    for directory, _subdirs, files in os.walk(output):
        for filename in files:
            path = Path(directory) / filename
            if path.name != "artifact_checksums.json" and path.name != "steps.jsonl":
                checksums[path.relative_to(output).as_posix()] = file_hash(path)
    write_json(output / "artifact_checksums.json", {"files": checksums,
               "code": {str(p.relative_to(root)).replace('\\', '/'): file_hash(p) for p in (
                   root / "src/asl_yolo/review532.py", root / "scripts/review_asl_group_532.py",
                   root / "scripts/review_asl29.py", root / "scripts/start_group_532_review.ps1")}})
    return stats


def prepare(root, dataset, output, workers=4, assess_only=False):
    manifest, rows = load_base(dataset)
    selected = [r for r in rows if eligible(r)]
    by_id = {r["sample_id"]: r for r in selected}
    helpers = load_helpers(root)
    signature = dict(base_fingerprint=manifest["fingerprint"], method=METHOD, selected_ids_sha256=digest(sorted(by_id)),
                     pipeline_sha256=file_hash(Path(__file__)),
                     machine_code_sha256=machine_code_fingerprint(Path(__file__).read_text(encoding="utf-8")),
                     asset_sha256=file_hash(root / "assets/hand_landmarker.task"),
                     helper_sha256=file_hash(root / "scripts/auto_review_asl29.py"),
                     mediapipe=importlib.metadata.version("mediapipe"))
    output.mkdir(parents=True, exist_ok=True)
    meta = output / "round.json"
    if meta.exists():
        assert_compatible_round(read_json(meta), signature, output)
    if not meta.exists():
        snapshot = output / "source_snapshot/review532.py"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(Path(__file__).read_bytes())
        write_json(meta, {"signature": signature, "created_at": utc(), "policy": "hybrid_sampling_with_human_exceptions",
                          "human_approval_fabricated": False, "base_dataset": str(dataset), "training_ready": False,
                          "parameters": {"release_iou": helpers.RELEASE_IOU, "minimum_dimension": helpers.RELEASE_MIN_DIMENSION,
                                         "release_padding": helpers.RELEASE_PADDING, "release_margin": helpers.RELEASE_MARGIN,
                                         "proposal_padding": .15, "representatives_per_stratum": 3}})
    evidence_path = output / "evidence.jsonl"
    evidence = load_review(evidence_path).get("annotations", {}) if evidence_path.exists() else {}
    for key, decision in evidence.items():
        if key not in by_id or decision["sha256"] != by_id[key]["sha256"] or decision.get("method") != METHOD:
            raise ValueError("Stale or unexpected automated evidence")
    pending = [r for r in selected if r["sample_id"] not in evidence]
    if assess_only and pending:
        raise ValueError(f"Automated QC is incomplete: {len(pending)} images")
    print(f"Eligible development: {len(selected)}; remaining automated checks: {len(pending)}", flush=True)
    event(output, "qc_started", eligible=len(selected), remaining=len(pending), base_fingerprint=manifest["fingerprint"])
    trackers = []
    def initialize():
        LOCAL.tracker = HandTracker(root / "assets/hand_landmarker.task")
        trackers.append(LOCAL.tracker)
    try:
        if pending:
            with evidence_path.open("a", encoding="utf-8", buffering=1) as stream:
                with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
                    for index, (row, decision) in enumerate(zip(pending, pool.map(lambda r: inspect_sample(dataset, r, helpers), pending)), 1):
                        key = row["sample_id"]
                        stream.write(json.dumps({"sample_id": key, "decision": decision}, allow_nan=False) + "\n")
                        evidence[key] = decision
                        if index % 250 == 0 or index == len(pending):
                            print(f"Checked {index}/{len(pending)}", flush=True)
                            event(output, "qc_progress", processed=index, total=len(pending))
    finally:
        for tracker in trackers:
            tracker.close()
    stats = make_packet(root, dataset, output, rows, evidence)
    event(output, "qc_packet_completed", **stats)
    return stats
