"""CLI độc lập notebook; không có lệnh chạy test/external/webcam."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
import uuid
from ..core import file_hash, read_json, write_json
from .config import WORK, load_training_config, preflight
from .reporting import analyze_validation, artifact_manifest, plot_distribution, regenerate_figures
from .runner import resume_probe, run_training
from .storage import RunLock, recover_run, unlock_after_verification, utc, valid_commits


def record_step(root: Path, command: list[str], status: str, result) -> None:
    """Lưu timeline thật và lỗi, không dùng số mẫu làm kết quả model."""
    report=root/'reports'/WORK
    report.mkdir(parents=True,exist_ok=True)
    event=dict(at_utc=utc(),argv=command,status=status,result=result)
    with (report/'steps.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,ensure_ascii=False)+'\n')
    with (report/'IMPORTANT_STEPS.md').open('a',encoding='utf-8') as stream:
        stream.write(f"\n## {event['at_utc']} — {status}\n\nLệnh: `python scripts/train_release.py {' '.join(command)}`\n\n```json\n{json.dumps(result,ensure_ascii=False,indent=2)}\n```\n")


def main(argv: list[str] | None = None) -> int:
    """Preflight, smoke, probe, baseline, resume, validation và phục hồi."""
    command=list(argv if argv is not None else sys.argv[1:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/asl_yolo11n_auto_v2_training.json'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight')
    p.add_argument('--data-only',action='store_true',help='Audit on CPU without pretending GPU checks passed')
    p.add_argument('--output',type=Path)
    p=sub.add_parser('train')
    p.add_argument('--mode',choices=['smoke','resume_probe','baseline'],required=True)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--result-file',type=Path)
    sub.add_parser('resume-probe')
    p=sub.add_parser('resume')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    for name in ['analyze','plots','status','recover','benchmark']:
        p=sub.add_parser(name)
        p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('unlock')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--old-job-stopped',action='store_true')
    args=parser.parse_args(command)
    root=Path(__file__).resolve().parents[3]
    # Package layout keeps script and src under the same root; CWD need not be old notebook state.
    cfg_path=args.config if args.config.is_absolute() else root/args.config
    try:
        cfg=load_training_config(cfg_path)
        if args.command=='preflight':
            output=args.output or root/'reports'/WORK/'verification'/('preflight_'+uuid.uuid4().hex[:8])
            result=preflight(root,cfg,output,require_gpu=not args.data_only)
            if result['data_ready']:
                meta=read_json(root/cfg['release']/'release.json')
                plot_distribution(meta['samples'],meta['classes'],output/'data_figures')
            summary={k:result.get(k) for k in ['data_ready','runtime_ready','fingerprint','counts','positive_support','background_counts','blockers']}
            summary['artifact']=str(output.resolve())
            record_step(root,command,'passed' if (args.data_only and result['data_ready']) or result['runtime_ready'] else 'blocked',summary)
            print(json.dumps(summary,ensure_ascii=False,indent=2))
            return 0 if (args.data_only and result['data_ready']) or result['runtime_ready'] else 2
        if args.command=='train':
            run=run_training(root,cfg,args.mode,stop_after=args.stop_after)
            result=dict(run=str(run.resolve()),state=read_json(run/'run.json'))
            if args.result_file:
                write_json(args.result_file,result)
        elif args.command=='resume':
            mode=read_json(args.run/'run.json')['mode']
            run=run_training(root,cfg,mode,resume_run=args.run,checkpoint=args.checkpoint)
            result=dict(run=str(run.resolve()),state=read_json(run/'run.json'))
        elif args.command=='resume-probe':
            result=resume_probe(root,cfg)
        elif args.command=='analyze':
            result=analyze_validation(root,args.run,cfg)
        elif args.command=='plots':
            with RunLock(args.run):
                regenerate_figures(args.run)
            result=dict(figures=str((args.run/'figures').resolve()))
        elif args.command=='status':
            commits,invalid=valid_commits(args.run)
            result=dict(state=read_json(args.run/'run.json'),valid_epochs=[c['epoch'] for c in commits],invalid=invalid,
                        latest_checkpoint=commits[-1]['checkpoint'] if commits else None,
                        lock=read_json(args.run/'.writer.lock') if (args.run/'.writer.lock').exists() else None,
                        partials=[str(p.relative_to(args.run)) for p in args.run.rglob('*.partial')])
        elif args.command=='recover':
            result=recover_run(args.run)
        elif args.command=='unlock':
            unlock_after_verification(args.run,args.old_job_stopped)
            result=dict(lock_released=True,operator_verified_old_job_stopped=True)
        elif args.command=='benchmark':
            from .benchmark import benchmark_forward
            result=benchmark_forward(root,args.run,cfg)
        else:
            raise ValueError('Unknown operation')
        record_step(root,command,'completed',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        result=dict(error=f'{type(exc).__name__}: {exc}')
        record_step(root,command,'failed',result)
        traceback.print_exc()
        return 1
