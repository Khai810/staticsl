"""Adapter release bất biến; không gọi manifest/review của prototype."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import random
import shutil
from pathlib import Path
import yaml
from ..core import digest, file_hash, read_json, read_rgb
from ..protocol import CLASSES_29, validate_annotation


def safe_path(root: Path, relative: str) -> Path:
    """Chặn đường dẫn thoát khỏi snapshot dữ liệu."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside release: {relative}')
    return path


def verify_leakage(root: Path, cfg: dict, samples: list[dict]) -> dict:
    """Đối chiếu metadata master, không mở ảnh holdout để chạy model."""
    base = root / cfg['base_dataset']
    for rel, expected in cfg['base_audit_sha256'].items():
        if file_hash(base / rel) != expected:
            raise ValueError(f'Base leakage evidence changed: {rel}')
    with (base / 'manifest.jsonl').open(encoding='utf-8') as stream:
        master = [json.loads(line) for line in stream if line.strip()]
    by_id = {r['sample_id']: r for r in master}
    ids = {r['sample_id'] for r in samples}
    groups: dict[tuple, list] = defaultdict(list)
    for row in master:
        for field in ('sha256', 'pixel_sha256', 'split_group'):
            if row.get(field):
                groups[(field, row[field])].append(row)
        if row.get('session_id'):
            groups[('known_session', row.get('member'), row['session_id'])].append(row)
    for sample in samples:
        row = by_id[sample['sample_id']]
        if (row['role'] != 'development' or row['exclusions'] or row['split'] != sample['split']
                or row['label'] != sample['label'] or row['class_id'] != sample['class_id']
                or row['output_sha256'] != sample['image_sha256']):
            raise ValueError(f"Unsafe membership: {sample['sample_id']}")
        for field in ('sha256', 'pixel_sha256', 'split_group'):
            if any(other['split'] != row['split'] for other in groups[(field, row[field])]):
                raise ValueError(f"Known {field} cross-split conflict: {sample['sample_id']}")
        if row.get('session_id'):
            if any(other['split'] != row['split'] for other in groups[('known_session', row.get('member'), row['session_id'])]):
                raise ValueError('Known session crosses split')
    with (base / 'audit/phash_candidates.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            pair = json.loads(line)
            if pair['cross_split'] and (pair['a'] in ids or pair['b'] in ids):
                raise ValueError('Unresolved cross-split pHash candidate; pHash alone is not proof of duplication')
    return dict(master_rows=len(master), accepted_rows=len(samples), conflicts=0,
                unknown_session_rows=sum(not by_id[s['sample_id']].get('session_id') for s in samples),
                scope='byte/pixel/known session groups and unresolved pHash candidates against source metadata; not signer independence')


def verify_release(root: Path, cfg: dict) -> dict:
    """Kiểm tra mapping, ảnh/label, split, checksum và leakage release."""
    release = root / cfg['release']
    for name, expected in cfg['release_metadata_sha256'].items():
        if file_hash(release / name) != expected:
            raise ValueError(f'Frozen release metadata changed: {name}')
    meta = read_json(release / 'release.json')
    samples = meta['samples']
    if (meta['kind'] != 'asl29_qc_run_release' or tuple(meta['classes']) != CLASSES_29
            or meta['fingerprint'] != cfg['fingerprint'] or digest(samples) != cfg['fingerprint']
            or not meta['training_ready'] or meta['evaluation_ready'] or not meta['run_split_frozen']
            or meta['review_kind'] != 'automated' or meta['human_review_claimed'] is not False):
        raise ValueError('Release identity/readiness/mapping/review mismatch')
    counts = dict(Counter(s['split'] for s in samples))
    if counts != cfg['expected_counts'] or counts != meta['counts']:
        raise ValueError(f'Release count mismatch: {counts}; do not resplit')
    if len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('Repeated sample ID')

    def check(row: dict) -> tuple[str, int]:
        split, sid = row['split'], row['sample_id']
        if split not in ('train', 'val') or row['class_id'] != meta['classes'].index(row['label']):
            raise ValueError('Invalid split or class ID')
        image, label = safe_path(release, row['image']), safe_path(release, row['label_file'])
        if (Path(row['image']).parts[:3] != ('draft', 'images', split) or image.stem != sid
                or Path(row['label_file']).parts[:3] != ('draft', 'labels', split) or label.stem != sid):
            raise ValueError('Unexpected release paths')
        if file_hash(image) != row['image_sha256'] or file_hash(label) != row['label_sha256']:
            raise ValueError(f'Image/label checksum mismatch: {sid}')
        boxes = []
        for line in label.read_text(encoding='ascii').splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] != str(row['class_id']):
                raise ValueError(f'Bad YOLO label: {sid}')
            boxes.append(list(map(float, fields[1:])))
        validate_annotation(row['label'], row['capture_type'], boxes)
        if len(boxes) > 1 or boxes != [[float(f'{v:.10f}') for v in b] for b in row['boxes']]:
            raise ValueError('Need exactly one positive GT or empty background')
        rgb = read_rgb(image)
        if rgb.ndim != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError('Unreadable image')
        return split, len(boxes)

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(check, samples))
    for split in ('train', 'val'):
        expected = ['./' + Path(s['image']).relative_to('draft').as_posix() for s in samples if s['split'] == split]
        if (release / 'draft' / f'{split}.txt').read_text().splitlines() != expected:
            raise ValueError('Frozen run list differs')
        for directory, field in [('images', 'image'), ('labels', 'label_file')]:
            actual = {p.relative_to(release).as_posix() for p in (release / 'draft' / directory / split).iterdir() if p.is_file()}
            if actual != {s[field] for s in samples if s['split'] == split}:
                raise ValueError('Extra/missing image or label')
    original_yaml = yaml.safe_load((release / 'draft/data.yaml').read_text())
    if original_yaml['names'] != dict(enumerate(CLASSES_29)) or 'test' in original_yaml:
        raise ValueError('Frozen YAML contains wrong mapping or test split')
    positives = {split: dict(Counter(s['label'] for s in samples if s['split'] == split and s['boxes'])) for split in ('train', 'val')}
    supported = [c for c in meta['classes'] if positives['val'].get(c, 0)]
    if positives['train'].get('nothing') != 22 or positives['val'].get('nothing', 0) != 0 or len(supported) != 28:
        raise ValueError('Support differs from authorized release')
    leakage = verify_leakage(root, cfg, samples)
    return dict(metadata=meta, samples=samples, fingerprint=meta['fingerprint'], counts=counts,
                positive_support=positives, supported_classes=supported,
                excluded_from_macro=[c for c in meta['classes'] if c not in supported],
                background_counts=dict(Counter(s['split'] for s in samples if not s['boxes'])),
                verified_images_and_labels=len(verified), leakage=leakage)


def prepare_dataset(release: Path, samples: list[dict], output: Path) -> Path:
    """Tạo adapter runtime bằng hardlink/copy, giữ nguyên tỷ lệ ảnh."""
    output.mkdir(parents=True, exist_ok=True)
    def materialize(row: dict) -> None:
        for field in ('image', 'label_file'):
            source = safe_path(release, row[field])
            target = output / Path(row[field]).relative_to('draft')
            target.parent.mkdir(parents=True, exist_ok=True)
            expected_hash = row['image_sha256' if field == 'image' else 'label_sha256']
            if target.exists():
                if not target.samefile(source) and file_hash(target) != expected_hash:
                    raise ValueError(f'Runtime adapter file differs: {target}')
                continue
            try:
                target.hardlink_to(source)
            except FileExistsError:
                if file_hash(target) != expected_hash:
                    raise ValueError('Concurrent adapter writer produced different contents')
            except OSError:
                shutil.copyfile(source, target)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(materialize,samples))
    for split in ('train', 'val'):
        paths = [(output / Path(s['image']).relative_to('draft')).resolve().as_posix() for s in samples if s['split'] == split]
        (output / f'{split}.txt').write_text('\n'.join(paths) + '\n', encoding='utf-8')
    target = output / 'data.yaml'
    target.write_text(yaml.safe_dump(dict(path=output.resolve().as_posix(), train='train.txt', val='val.txt',
                                        names=dict(enumerate(CLASSES_29))), sort_keys=False), encoding='utf-8')
    return target


def select_smoke(samples: list[dict], seed: int) -> list[dict]:
    """Giữ split, lấy tối đa 4 train/2 val mỗi lớp và capture type."""
    groups: dict[tuple, list] = defaultdict(list)
    for row in samples:
        groups[(row['split'], row['label'], row['capture_type'])].append(row)
    rng, selected = random.Random(seed), []
    for key, group in sorted(groups.items()):
        candidates = sorted(group, key=lambda s: s['sample_id'])
        selected.extend(rng.sample(candidates, min(len(candidates), 4 if key[0] == 'train' else 2)))
    return sorted(selected, key=lambda s: s['sample_id'])


def check_loaders(data_yaml: Path, cfg: dict, augmentation: dict) -> dict:
    """Đọc batch thật, kiểm tra train augmentation và val letterbox."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    data = yaml.safe_load(data_yaml.read_text())
    hyp = get_cfg(overrides={**cfg['train_args'], **augmentation, 'imgsz': cfg['imgsz']})
    report = {}
    for split in ('train', 'val'):
        dataset = YOLODataset(img_path=str(data_yaml.parent / f'{split}.txt'), imgsz=cfg['imgsz'],
                              batch_size=cfg['batch'], augment=split == 'train', hyp=hyp, rect=False,
                              data=data, cache=False, prefix=f'{split}: ')
        expected = (data_yaml.parent / f'{split}.txt').read_text().splitlines()
        if {Path(p).resolve() for p in dataset.im_files} != set(map(Path, expected)):
            raise ValueError('Framework removed/added samples')
        if len(dataset.labels) != len(expected):
            raise ValueError('Loader labels missing')
        batch = dataset.collate_fn([dataset[i] for i in range(min(4, len(dataset)))])
        if batch['img'].shape[1] != 3 or not torch.isfinite(batch['img']).all():
            raise ValueError('Loader produced invalid batch')
        report[split] = dict(images=len(dataset), batch_shape=list(batch['img'].shape),
                             augment=dataset.augment, transforms=str(dataset.transforms),
                             actual_transform_tree=audit_transforms(dataset.transforms,augmentation,split=='train'),
                             positive_boxes=sum(len(x['cls']) for x in dataset.labels),
                             backgrounds=sum(not len(x['cls']) for x in dataset.labels))
    if report['val']['augment'] or 'RandomPerspective' in report['val']['transforms']:
        raise ValueError('Validation augmentation must be off')
    return report


def audit_transforms(transform, augmentation: dict, training: bool) -> list[dict]:
    """Xác minh xác suất/giá trị thực của transform, kể cả Albumentations ngầm."""
    records=[]
    def visit(t) -> None:
        name=type(t).__name__
        record={'type':name}
        for key in ('p','degrees','translate','scale','shear','perspective','direction','hgain','sgain','vgain','bgr'):
            if hasattr(t,key): record[key]=getattr(t,key)
        if name=='Albumentations':
            active=getattr(t,'transform',None)
            transforms=list(getattr(active,'transforms',[])) if active is not None else []
            record['configured_transforms']=[str(x) for x in transforms]
            if transforms: raise ValueError('Unexpected implicit Albumentations')
        if name=='RandomPerspective':
            for key in ('degrees','translate','scale','shear','perspective'):
                if getattr(t,key)!=augmentation[key]: raise ValueError(f'Actual transform changed {key}')
        probabilities={'Mosaic':'mosaic','CopyPaste':'copy_paste','MixUp':'mixup','CutMix':'cutmix'}
        if name in probabilities and t.p!=augmentation[probabilities[name]]:
            raise ValueError('Actual augmentation probability changed')
        if name=='RandomFlip' and t.p!=augmentation['fliplr' if t.direction=='horizontal' else 'flipud']:
            raise ValueError('Unexpected image flip')
        if name=='RandomHSV' and any(getattr(t,k)!=augmentation[v] for k,v in [('hgain','hsv_h'),('sgain','hsv_s'),('vgain','hsv_v')]):
            raise ValueError('Unexpected HSV augmentation')
        if not training and name not in ('Compose','LetterBox','Format'):
            raise ValueError('Unexpected validation transform')
        records.append(record)
        for child in getattr(t,'transforms',[]): visit(child)
    visit(transform)
    return records
