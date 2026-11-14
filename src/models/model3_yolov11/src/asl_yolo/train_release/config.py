"""Cổng train mới; metadata release không bao giờ bị bật allow_training."""
from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any
import uuid
import yaml
from ..core import digest, file_hash, read_json, versions, write_json
from .dataset import check_loaders, prepare_dataset, verify_release
from .storage import publish_bytes, utc

WORK = 'asl_yolo11n_auto_v2_training_20261113'
FINGERPRINT = 'fbb9d072079f9a08d42e953d5927b232f64497f0d4d3d3e8e9309e045772efdf'
ASSET_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def load_training_config(path: Path) -> dict[str, Any]:
    """Đọc cấu hình và khóa đúng release, YOLO11n, final test."""
    cfg = read_json(path)
    if (cfg.get('kind') != 'asl_auto_v2_training' or cfg.get('schema_version') != 1
            or cfg.get('fingerprint') != FINGERPRINT or cfg.get('pretrained_sha256') != ASSET_SHA256
            or cfg.get('model') != 'assets/yolo11n.pt'
            or cfg.get('allow_final_test') is not False or cfg.get('require_cuda') is not True
            or cfg.get('expected_counts') != {'train':9042,'val':3909}
            or cfg.get('imgsz') != 320 or cfg.get('seed') != 42 or cfg.get('freeze') != 0
            or cfg.get('batch') not in (8,4,2,1) or cfg.get('smoke_epochs') != 3
            or cfg.get('baseline_epochs') != 50 or cfg.get('ultralytics') != '8.4.178'):
        raise ValueError('Config violates authorized release/baseline protocol')
    if cfg['monitor_confidence'] != .25 or cfg['nms_iou'] != .7 or cfg['sweep_grid'] != [round(i/20,2) for i in range(1,20)]:
        raise ValueError('Metric/threshold protocol changed')
    args = cfg['train_args']
    if args['amp'] is not False or args['patience'] != 0 or args['workers'] != 0 or args['optimizer'] != 'SGD':
        raise ValueError('Baseline AMP/early-stopping/loader/optimizer mismatch')
    if not isinstance(cfg['allow_training'], bool):
        raise ValueError('allow_training must be boolean')
    return cfg


def environment() -> dict:
    """Ghi phiên bản thực tế và GPU đang dùng."""
    import torch
    packages = versions()
    for name in ('matplotlib','Pillow','PyYAML','psutil'):
        packages[name] = importlib.metadata.version(name)
    return dict(versions=packages, installed_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                python_executable=sys.executable, platform=platform.platform(),
                cuda=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)


def source_hashes(root: Path) -> dict[str, str]:
    """Khóa code chạy thực, dùng hash khi checkout không có Git."""
    files = sorted((root/'src/asl_yolo').rglob('*.py'))
    files += [root/'scripts/train_release.py']
    return {p.relative_to(root).as_posix():file_hash(p) for p in files}


def resolve_train_args(root: Path, cfg: dict, data_yaml: Path, run: Path, epochs: int) -> dict:
    """Resolve mọi mặc định trước khi tạo trainer."""
    from ultralytics.cfg import get_cfg
    aug = yaml.safe_load((root/cfg['release']/'augmentation_baseline.yaml').read_text())
    overrides = dict(cfg['train_args'], **aug)
    overrides.update(model=str((root/cfg['model']).resolve()), data=str(data_yaml.resolve()),
                     imgsz=cfg['imgsz'], batch=cfg['batch'], seed=cfg['seed'], freeze=cfg['freeze'],
                     epochs=epochs, device=0, save_dir=str((run/'framework').resolve()),
                     project=str(run.resolve()), name='framework', exist_ok=True,
                     conf=.001, iou=cfg['nms_iou'], max_det=300, task='detect', mode='train')
    return vars(get_cfg(overrides=overrides))


def preflight(root: Path, cfg: dict, output: Path, require_gpu: bool = True) -> dict:
    """Kiểm tra đủ dữ liệu/loader trước khi báo cổng GPU và storage."""
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    result = dict(at_utc=utc(), fingerprint=cfg['fingerprint'], environment=env,
                  data_ready=False, runtime_ready=False, training_or_model_evaluation_run=False, blockers=[])
    try:
        print('PREFLIGHT: verifying frozen release images, labels and source leakage evidence',flush=True)
        checked = verify_release(root, cfg)
        result.update({k:v for k,v in checked.items() if k not in ('metadata','samples')})
        if file_hash(root/cfg['model']) != cfg['pretrained_sha256']:
            raise ValueError('Pretrained COCO asset checksum mismatch')
        if env['versions']['ultralytics'] != cfg['ultralytics']:
            raise ValueError('Adapter requires Ultralytics 8.4.178; do not upgrade silently')
        print('PREFLIGHT: dataset verified; loading COCO asset',flush=True)
        from ultralytics import YOLO
        asset = YOLO(str(root/cfg['model']))
        if asset.task != 'detect' or asset.model.yaml.get('scale') != 'n' or len(asset.names) != 80:
            raise ValueError('Expected original COCO YOLO11n detection asset')
        result['pretrained_asset_loadable'] = True
        cache = root/'.cache/train_release'/cfg['fingerprint'][:12]/'full'
        print('PREFLIGHT: preparing separate runtime adapter',flush=True)
        data_yaml = prepare_dataset(root/cfg['release'], checked['samples'], cache)
        augmentation = yaml.safe_load((root/cfg['release']/'augmentation_baseline.yaml').read_text())
        print('PREFLIGHT: checking actual train/validation loaders',flush=True)
        result['loader_checks'] = check_loaders(data_yaml, cfg, augmentation)
        result.update(data_ready=True, augmentation=augmentation, data_yaml=str(data_yaml.resolve()),
                      review_kind='automated', human_review_claimed=False,
                      annotation_quality=checked['metadata']['annotation_quality'])
        # Runtime-specific checks follow data audit, so CPU still yields useful evidence.
        if require_gpu and not env['cuda']:
            result['blockers'].append('PyTorch has no CUDA GPU; smoke/resume/baseline not executed')
        if env['cuda'] and env['gpu_total_bytes'] < cfg['min_gpu_bytes']:
            result['blockers'].append('GPU VRAM below configured minimum; inspect before training')
        runs = Path(cfg['runs_root'])
        if not runs.is_absolute():
            runs = root/runs
        runs.mkdir(parents=True, exist_ok=True)
        if Path('/content').is_dir() and not runs.resolve().is_relative_to(Path('/content/drive').resolve()):
            result['blockers'].append('Colab runs_root must be on mounted Drive, not ephemeral /content')
        result['storage'] = dict(runs_root=str(runs.resolve()), free_bytes=shutil.disk_usage(runs).free,
                                 reserve_bytes=cfg['min_storage_bytes'],
                                 contract='per-epoch verified generation; Drive rename atomicity not assumed')
        if result['storage']['free_bytes'] < cfg['min_storage_bytes']:
            result['blockers'].append('Insufficient free durable storage')
        probe = runs/('.storage_probe_'+uuid.uuid4().hex)
        publish_bytes(probe, b'ASL durable storage readback probe\n')
        probe.unlink()
        result['storage']['write_readback_passed'] = True
        result['gpu_requirement_checked'] = require_gpu
        result['runtime_ready'] = env['cuda'] and not result['blockers']
        write_json(output/'resolved_train_args.json', resolve_train_args(root,cfg,data_yaml,runs/'PREFLIGHT_ONLY',cfg['baseline_epochs']))
        write_json(output/'class_mapping.json', checked['metadata']['classes'])
        write_json(output/'source_hashes.json', source_hashes(root))
    except Exception as exc:
        result['blockers'].append(f'{type(exc).__name__}: {exc}')
        write_json(output/'preflight.json', result)
        raise
    write_json(output/'preflight.json', result)
    return result
