"""Chia dataset ZIP để vận chuyển dưới100MiB; giữ nguyên byte và SHA dataset gốc."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def split_archive(source: Path, output: Path, part_bytes: int = 80*1024*1024) -> list[dict]:
    """Ghi từng phần và kiểm tra nối byte; không tạo dataset/split mới."""
    if part_bytes<1 or part_bytes>=100*1024*1024:
        raise ValueError('Part size must be positive and below100MiB')
    output.mkdir(parents=True,exist_ok=True)
    parts=[]
    with source.open('rb') as stream:
        index=1
        while block:=stream.read(part_bytes):
            name=f'{source.name}.part{index:03d}'
            target=output/name
            expected=hashlib.sha256(block).hexdigest()
            if target.exists():
                if sha256(target)!=expected: raise ValueError('Existing upload part differs')
            else:
                with target.open('xb') as dst: dst.write(block)
            if sha256(target)!=expected: raise IOError('Part verification failed')
            parts.append(dict(file=name,bytes=len(block),sha256=expected))
            index+=1
    joined=hashlib.sha256()
    for part in parts:
        with (output/part['file']).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): joined.update(block)
    if joined.hexdigest()!=sha256(source): raise IOError('Joined checksum differs from original ZIP')
    return parts


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.bundle.read_text(encoding='utf-8'))
    source=args.bundle.parent/manifest['dataset']['file']
    if sha256(source)!=manifest['dataset']['sha256']: raise ValueError('Original dataset archive differs')
    manifest['dataset']['parts']=split_archive(source,args.output)
    manifest['upload_status']='prepared chunked transport; verify Drive upload metadata separately'
    (args.output/'bundle.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output.resolve()),parts=manifest['dataset']['parts'],
                         original_dataset_sha256=manifest['dataset']['sha256']),indent=2))


if __name__=='__main__': main()
