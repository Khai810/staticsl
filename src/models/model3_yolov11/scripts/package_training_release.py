"""Đóng gói source và ONLY accepted train/val; không ảnh test/Bao/pending."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asl_yolo.core import file_hash, write_json
from asl_yolo.train_release.config import WORK, load_training_config
from asl_yolo.train_release.dataset import verify_release


def make_zip(output: Path, paths: list[Path], manifest_name: str) -> dict:
    """Không ghi đè bundle cũ; xác minh nội dung archive sau khi ghi."""
    records={}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(set(paths)):
            if not path.is_file(): raise FileNotFoundError(path)
            name=path.relative_to(ROOT).as_posix()
            archive.write(path,name)
            records[name]=file_hash(path)
        archive.writestr(manifest_name,json.dumps(dict(files=records),indent=2))
    with zipfile.ZipFile(output) as archive:
        for name,expected in records.items():
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
            if h.hexdigest()!=expected: raise IOError(f'Archive mismatch: {name}')
    return dict(file=output.name,bytes=output.stat().st_size,sha256=file_hash(output),entries=len(records))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/asl_yolo11n_auto_v2_training.json')
    args=parser.parse_args()
    cfg=load_training_config(args.config)
    checked=verify_release(ROOT,cfg)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'exports'/WORK/stamp
    output.mkdir(parents=True,exist_ok=False)
    source=sorted((ROOT/'src/asl_yolo').rglob('*.py'))
    source += sorted((ROOT/'tests').glob('*.py'))
    source += [ROOT/'scripts'/name for name in ['train_release.py','package_training_release.py','build_training_notebook.py']]
    source += [ROOT/name for name in ['pyproject.toml','requirements-training-colab.txt','configs/asl_yolo11n_auto_v2_training.json',
        'notebooks/03_asl_auto_v2_training.ipynb','assets/yolo11n.pt']]
    source += [ROOT/'reports'/WORK/name for name in ['IMPLEMENTATION_PLAN.md','SKELETON.md','CODE_GUIDE.md','REPORT.md','IMPORTANT_STEPS.md']]
    release=ROOT/cfg['release']
    dataset=[release/name for name in list(cfg['release_metadata_sha256'])+['draft/data.yaml','draft/train.txt','draft/val.txt']]
    dataset += [release/s[field] for s in checked['samples'] for field in ['image','label_file']]
    dataset += [ROOT/cfg['base_dataset']/name for name in cfg['base_audit_sha256']]
    source_result=make_zip(output/'source.zip',source,'source_bundle_manifest.json')
    data_result=make_zip(output/'dataset.zip',dataset,'dataset_bundle_manifest.json')
    result=dict(at_utc=datetime.now(timezone.utc).isoformat(),fingerprint=cfg['fingerprint'],counts=checked['counts'],
                source=source_result,dataset=data_result,
                scope='Accepted train/val images and immutable metadata, master metadata/pHash evidence only. No holdout images.',
                upload_status='local package only; not assumed uploaded to Drive')
    write_json(output/'bundle.json',result)
    write_json(ROOT/'reports'/WORK/'verification/package.json',dict(directory=str(output.resolve()),**result))
    print(json.dumps(dict(directory=str(output),**result),indent=2))


if __name__=='__main__': main()
