from __future__ import annotations

from collections import Counter, defaultdict
import csv
import random
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from .core import digest, file_hash, read_json, read_rgb, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Extraction target exists: {destination}")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if sum(i.file_size for i in entries) > 20 * 1024**3:
            raise ValueError("Archive exceeds 20 GiB uncompressed safety limit")
        seen = set()
        for i in entries:
            p = PurePosixPath(i.filename.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts or any(":" in part for part in p.parts) or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member: {i.filename}")
            key = str(p).casefold()
            if key in seen:
                raise ValueError(f"Duplicate ZIP destination: {i.filename}")
            seen.add(key)
        if z.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        destination.mkdir(parents=True)
        for i in entries:
            target = destination.joinpath(*PurePosixPath(i.filename.replace("\\", "/")).parts)
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(i) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out)


def acquire_dataset(cfg, env):
    root = Path(env["root"])
    archive = root / cfg["dataset_zip"]
    if not archive.is_file():
        # Explicit, documented alias for the supplied Kaggle ZIP; never select arbitrary ZIPs.
        alternate = root / "data/raw/archive.zip"
        if alternate.is_file():
            archive = alternate
            print("Using existing data/raw/archive.zip (original retained)")
        elif not cfg["allow_download"]:
            raise FileNotFoundError(f"Missing {archive}; upload ZIP or enable allow_download")
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            # Kaggle auth is handled by its own UI/config or Colab secret, never logged here.
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                with tempfile.TemporaryDirectory(dir=archive.parent) as temp:
                    api.dataset_download_files(cfg["dataset_slug"], path=temp, quiet=True, unzip=False)
                    found = list(Path(temp).glob("*.zip"))
                    if len(found) != 1:
                        raise RuntimeError("Expected exactly one downloaded ZIP")
                    if not zipfile.is_zipfile(found[0]):
                        raise ValueError("Kaggle response is not ZIP")
                    shutil.move(str(found[0]), archive)
            except Exception:
                raise RuntimeError("Kaggle download unavailable. Configure Kaggle credentials outside the repo or Colab secret KAGGLE_API_TOKEN; alternatively place the original ZIP in data/raw.") from None
    archive_hash = file_hash(archive)
    target = Path(env["cache"]) / "extracted" / archive_hash[:16]
    marker = target / ".complete.json"
    if target.exists():
        if not marker.is_file() or read_json(marker)["sha256"] != archive_hash:
            raise RuntimeError(f"Incomplete/incompatible extraction: {target}. Use a new cache path; original is untouched.")
    else:
        safe_extract(archive, target)
        write_json(marker, {"sha256": archive_hash})
    result = {"archive": str(archive), "sha256": archive_hash, "extracted": str(target), "slug": cfg["dataset_slug"]}
    write_json(Path(env["work"]) / "dataset.json", result)
    return result


def duplicate_groups(rows, distance=2):
    """Exact hashes + transitive pHash groups, using a Hamming-distance BK tree."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        parents[find(b)] = find(a)
    exact, tree = {}, None
    for i, row in enumerate(rows):
        if row["sha256"] in exact:
            union(i, exact[row["sha256"]])
        exact[row["sha256"]] = i
        value = int(row["phash"], 16)
        if tree is None:
            tree = [value, i, {}]
            continue
        stack = [tree]
        while stack:
            node = stack.pop()
            d = (value ^ node[0]).bit_count()
            if d <= distance:
                union(i, node[1])
            stack.extend(child for edge, child in node[2].items() if d - distance <= edge <= d + distance)
        node = tree
        while True:
            d = (value ^ node[0]).bit_count()
            if d == 0:
                break
            if d not in node[2]:
                node[2][d] = [value, i, {}]
                break
            node = node[2][d]
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    for members in groups.values():
        group = min(rows[i]["sample_id"] for i in members)
        conflict = len({rows[i]["class_name"] for i in members}) > 1
        for i in members:
            rows[i]["duplicate_group"] = group
            rows[i]["label_conflict"] = conflict
    return rows


def audit_dataset(dataset, cfg, env):
    import imagehash
    from PIL import Image
    base = Path(dataset["extracted"])
    rows, corrupt, annotations = [], [], []
    actual = Counter()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == ".complete.json":
            continue
        rel = path.relative_to(base).as_posix()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            annotations.append(rel)
            continue
        label = path.parent.name
        actual[label] += 1
        try:
            rgb = read_rgb(path)
            h, w = rgb.shape[:2]
            if min(h, w) < 2:
                raise ValueError("Image too small")
            phash = str(imagehash.phash(Image.fromarray(rgb)))
            sha = file_hash(path)
            existing = {"valid": "val", "validation": "val", "train": "train", "val": "val", "test": "test"}
            declared = {existing[p.lower()] for p in path.relative_to(base).parts[:-1] if p.lower() in existing}
            if len(declared) > 1:
                raise ValueError("Ambiguous existing split directories")
            rows.append({"sample_id": digest(rel)[:20], "path": rel, "sha256": sha, "phash": phash,
                         "class_name": label.upper(), "original_label": label, "width": w, "height": h,
                         "source_split": next(iter(declared), None), "status": "valid"})
        except Exception as exc:
            corrupt.append({"path": rel, "error": str(exc)})
    missing = set(cfg["classes"]) - {r["class_name"] for r in rows}
    if missing:
        raise ValueError(f"Configured classes missing: {sorted(missing)}; discovered labels: {dict(actual)}")
    duplicate_groups(rows, cfg["near_duplicate_distance"])
    report = {"archive_sha256": dataset["sha256"], "actual_counts": dict(actual), "valid_images": len(rows),
              "corrupt": corrupt, "annotation_candidates": annotations, "selected_classes": cfg["classes"],
              "excluded_classes": sorted(set(actual) - {r["original_label"] for r in rows if r["class_name"] in cfg["classes"]}),
              "actual_to_canonical": {r["original_label"]: r["class_name"] for r in rows},
              "exact_duplicate_excess": len(rows) - len({r["sha256"] for r in rows}),
              "duplicate_groups": len({r["duplicate_group"] for r in rows}),
              "conflicting_samples": sum(r["label_conflict"] for r in rows),
              "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in rows)), "rows": rows}
    write_json(Path(env["report_dir"]) / "audit.json", report)
    import matplotlib.pyplot as plt
    labels = sorted(actual)
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for ax, label in zip(axes.flat, labels):
        candidates = [r for r in rows if r["original_label"] == label]
        if candidates:
            ax.imshow(read_rgb(base / candidates[0]["path"]))
        ax.set_title(f"{label}: {actual[label]}")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(env["report_dir"]) / "samples.png", dpi=120)
    plt.close(fig)
    print({k: v for k, v in report.items() if k not in ("rows", "dimensions", "corrupt")})
    return report


def assign_splits(rows, ratios, seed):
    """Assign indivisible groups, approximately stratified by label count."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("split_group", row["duplicate_group"])].append(row)
    counts = Counter(r["class_name"] for r in rows)
    current = {s: Counter() for s in SPLITS}
    rng = random.Random(seed)
    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    # Honor declared splits first so the deficit calculation accounts for them.
    fixed, free = [], []
    for group in groups:
        declared = {r["source_split"] for r in group if r.get("source_split")}
        if len(declared) > 1:
            raise ValueError("Duplicate/signer group crosses existing splits; resolve source leakage first")
        (fixed if declared else free).append((group, next(iter(declared), None)))
    for group, declared in fixed + free:
        gc = Counter(r["class_name"] for r in group)
        if declared:
            chosen = declared
        else:
            chosen = max(SPLITS, key=lambda s: sum(n * (counts[c] * ratios[SPLITS.index(s)] - current[s][c]) / max(counts[c] * ratios[SPLITS.index(s)], 1) for c, n in gc.items()))
        for row in group:
            row["split"] = chosen
        current[chosen].update(gc)
    for label in counts:
        if any(current[s][label] == 0 for s in SPLITS):
            raise ValueError(f"Insufficient independent groups to populate all splits for {label}. Increase max_per_class or review duplicate grouping; use a new work_name.")
    return rows


def build_or_load_manifest(audit, cfg, env):
    selected = [dict(r) for r in audit["rows"] if r["class_name"] in cfg["classes"]]
    metadata_path = cfg.get("metadata_csv")
    metadata_hash = None
    if metadata_path:
        path = Path(env["root"]) / metadata_path
        metadata_hash = file_hash(path)
        with path.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        by_path = {r["path"]: r for r in records}
        if len(by_path) != len(records):
            raise ValueError("Duplicate paths in metadata_csv")
        # Merge subject/session and duplicate relationships with union-find.
        groups = {}
        def find(x):
            groups.setdefault(x, x)
            if groups[x] != x:
                groups[x] = find(groups[x])
            return groups[x]
        for row in selected:
            m = by_path.get(row["path"])
            if not m or not m.get("group"):
                raise ValueError("metadata_csv requires path,group for every selected image")
            a, b = find("d:" + row["duplicate_group"]), find("m:" + m["group"])
            groups[a] = b
            if m.get("split"):
                if m["split"] not in SPLITS or row["source_split"] not in (None, m["split"]):
                    raise ValueError("Invalid/conflicting metadata split")
                row["source_split"] = m["split"]
        for row in selected:
            row["split_group"] = find("d:" + row["duplicate_group"])
    else:
        for row in selected:
            row["split_group"] = row["duplicate_group"]
    signature = {"schema": 1, "archive": audit["archive_sha256"], "classes": cfg["classes"], "seed": cfg["seed"],
                 "split": cfg["split"], "max_per_class": cfg["max_per_class"], "near_distance": cfg["near_duplicate_distance"],
                 "box_padding": cfg["box_padding"], "metadata_hash": metadata_hash,
                 "audit_fingerprint": digest([(r["path"], r["sha256"], r["phash"]) for r in selected])}
    path = Path(env["work"]) / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if digest(manifest["rows"]) != manifest["fingerprint"]:
            raise ValueError("Manifest was edited; restore original or use a new work_name")
        if manifest["signature"] != signature:
            raise ValueError("Dataset/config changed. Choose a new work_name; existing review/splits remain untouched.")
        return manifest
    rng = random.Random(cfg["seed"])
    chosen = set()
    for label in cfg["classes"]:
        candidates = sorted(r["sample_id"] for r in selected if r["class_name"] == label)
        rng.shuffle(candidates)
        chosen.update(candidates[:cfg["max_per_class"]])
    # Selection is capped per class; grouping still uses ALL images before selection.
    rows = [r for r in selected if r["sample_id"] in chosen]
    assign_splits(rows, cfg["split"], cfg["seed"])
    for row in rows:
        row["class_id"] = cfg["classes"].index(row["class_name"])
        row["status"] = "pending_review"
    manifest = {"signature": signature, "fingerprint": digest(rows), "classes": cfg["classes"], "rows": rows,
                "split_limitation": "Subject/session grouping supplied" if metadata_path else "No verified signer/session metadata: random grouped split does NOT establish unseen-signer generalization."}
    write_json(path, manifest)
    write_json(Path(env["work"]) / "class_mapping.json", cfg["classes"])
    print("Split counts:", dict(Counter((r["split"], r["class_name"]) for r in rows)))
    print(manifest["split_limitation"])
    return manifest
