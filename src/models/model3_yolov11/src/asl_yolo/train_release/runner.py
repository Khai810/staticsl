"""New run, OOM attempt mới và resume cùng thí nghiệm trên CUDA."""
from __future__ import annotations
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import yaml
from ..core import digest, file_hash, read_json, write_json
from ..training import assert_mapping
from .config import WORK, environment, preflight, resolve_train_args, source_hashes
from .dataset import prepare_dataset, select_smoke
from .engine import trainer_class
from .storage import ControlledStop, RunLock, utc, valid_commits, verify_resume


class Tee:
    def __init__(self, terminal, log):
        self.terminal,self.log = terminal,log
    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return False


def runs_root(root: Path, cfg: dict) -> Path:
    path = Path(cfg['runs_root'])
    return path if path.is_absolute() else root/path


def require_gate(root: Path, cfg: dict, name: str) -> dict:
    """Baseline chỉ chạy sau bằng chứng GPU thật, cùng code/release."""
    gate = read_json(runs_root(root,cfg)/'gates'/f'{name}.json')
    if (not gate['passed'] or not gate['cuda'] or gate['fingerprint'] != cfg['fingerprint']
            or gate['source_hashes'] != source_hashes(root)):
        raise ValueError(f'{name} evidence missing or belongs to other code/release')
    for path,sha in gate['artifacts'].items():
        if file_hash(path) != sha:
            raise ValueError(f'{name} evidence artifact changed')
    return gate


def verify_trained_artifacts(run: Path, cfg: dict, samples: list[dict], release: Path) -> dict:
    """Load model thật và một prediction kỹ thuật sau train; không test."""
    from ultralytics import YOLO
    import torch
    commits,invalid = valid_commits(run)
    if invalid or not commits:
        raise ValueError('Incomplete run journal')
    history = read_json(run/'metrics/history.json')
    if [h['epoch'] for h in history] != list(range(1,len(commits)+1)):
        raise ValueError('History is not contiguous/unique')
    best = run/'checkpoints/best_macro_f1.pt'
    model = YOLO(str(best))
    assert_mapping(model.names,read_json(run/'class_mapping.json'))
    sample = next(s for s in samples if s['split']=='val')
    result = model.predict(str(release/sample['image']),imgsz=cfg['imgsz'],conf=.25,iou=cfg['nms_iou'],
                           device=0,verbose=False)[0]
    if not torch.isfinite(result.boxes.data).all():
        raise ValueError('Nonfinite model predictions')
    ckpt = torch.load(run/commits[-1]['checkpoint'],map_location='cpu',weights_only=False)
    if not ckpt['optimizer']['state'] or ckpt['epoch']+1 != len(history):
        raise ValueError('Resume checkpoint lost optimizer/epoch')
    result = dict(at_utc=utc(),passed=True,epoch_count=len(history),model_load_and_predict=True,
                  positive_validation_sample=sample['sample_id'],checkpoint_sha256=file_hash(best),
                  optimizer_present=True,optimizer_state_entries=len(ckpt['optimizer']['state']),
                  native_best_role='framework/weights/best.pt uses Ultralytics fitness',
                  inference_role='checkpoints/best_macro_f1.pt uses supported macro F1',
                  resume_role='commits/* references full checkpoints; last_resume.pt is a verified convenience alias')
    write_json(run/'model_load.json',result)
    del model,ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_training(root: Path, cfg: dict, mode: str, resume_run: Path | None = None, checkpoint: Path | None = None, stop_after: int | None = None) -> Path:
    """Train CUDA; OOM mới tạo run mới từ pretrained, resume không đổi config."""
    import torch
    if mode not in ('smoke','resume_probe','baseline') or not cfg['allow_training'] or cfg['allow_final_test']:
        raise ValueError('Training gates/mode invalid; final test always disabled')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. No CPU training fallback; run preflight --data-only locally')
    if resume_run is not None and (checkpoint is None or stop_after is not None):
        raise ValueError('Resume needs explicit checkpoint and no new stopping configuration')
    verification = root/'reports'/WORK/'verification'/('preflight_'+uuid.uuid4().hex[:8])
    pf = preflight(root,cfg,verification,require_gpu=True)
    if not pf['runtime_ready']:
        raise RuntimeError(f"Preflight blocked: {pf['blockers']}")
    release = root/cfg['release']
    meta = read_json(release/'release.json')
    all_samples = meta['samples']
    selected = all_samples if mode == 'baseline' else select_smoke(all_samples,cfg['seed'])
    if mode == 'baseline' and resume_run is None:
        require_gate(root,cfg,'smoke')
        require_gate(root,cfg,'resume')
    total_epochs = cfg['baseline_epochs'] if mode == 'baseline' else cfg['smoke_epochs'] if mode == 'smoke' else 4
    if stop_after is not None and not 0 < stop_after < total_epochs:
        raise ValueError('Controlled stop must precede total epochs')
    attempts = []
    while True:
        run = Path(resume_run) if resume_run else runs_root(root,cfg)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+mode+'_'+uuid.uuid4().hex[:8])
        if resume_run is None:
            run.mkdir(parents=True,exist_ok=False)
        with RunLock(run):
            resumed = None
            if resume_run:
                resumed = verify_resume(run,Path(checkpoint),cfg)
                state = read_json(run/'run.json')
                if state['mode'] != mode or state['total_epochs'] != total_epochs:
                    raise ValueError('Resume mode/epoch target changed')
                previous_env,current_env = read_json(run/'environment.json'),environment()
                if any(previous_env['versions'][k] != current_env['versions'][k] for k in ('python','torch','torchvision','ultralytics','numpy')):
                    raise ValueError('Runtime versions changed; same-version resume required')
                if read_json(run/'source_hashes.json') != source_hashes(root):
                    raise ValueError('Training source changed since checkpoint')
                if read_json(run/'samples.json') != selected:
                    raise ValueError('Frozen subset changed')
            else:
                state = dict(run_id=run.name,mode=mode,status='starting',started_utc=utc(),
                             total_epochs=total_epochs,fingerprint=cfg['fingerprint'],
                             config_sha256=digest(cfg),pretrained_sha256=cfg['pretrained_sha256'],
                             initialization='fresh verified COCO YOLO11n',prior_oom_attempts=attempts,
                             review_kind='automated',human_review_claimed=False,allow_final_test=False)
                for name,value in [('config.json',cfg),('class_mapping.json',meta['classes']),('environment.json',environment()),
                                   ('samples.json',selected),('source_hashes.json',source_hashes(root)),
                                   ('dataset_reference.json',{k:v for k,v in meta.items() if k!='samples'}),('preflight.json',pf)]:
                    write_json(run/name,value)
            state.update(status='running',updated_utc=utc())
            write_json(run/'run.json',state)
            adapter = root/'.cache/train_release'/cfg['fingerprint'][:12]/('full' if mode=='baseline' else 'smoke_seed42')
            data_yaml = prepare_dataset(release,selected,adapter)
            args = resolve_train_args(root,cfg,data_yaml,run,total_epochs)
            if resume_run:
                args.update(model=str(Path(checkpoint).resolve()),resume=str(Path(checkpoint).resolve()))
            context = dict(run=str(run),cfg=cfg,classes=meta['classes'],
                           val_samples=[s for s in selected if s['split']=='val'],supported=pf['supported_classes'],
                           augmentation=pf['augmentation'],source_hashes=source_hashes(root),
                           resume_ckpt=resumed,stop_after=stop_after)
            started = time.perf_counter()
            trainer = None
            oom = False
            try:
                log_path = run/'logs'/('session_'+uuid.uuid4().hex[:8]+'.log')
                log_path.parent.mkdir(parents=True,exist_ok=True)
                with log_path.open('w',encoding='utf-8',buffering=1) as log, redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                    trainer = trainer_class(context)(overrides=args)
                    trainer.train()
                commits,invalid = valid_commits(run)
                if invalid or len(commits) != total_epochs:
                    raise ValueError('Unexpected early termination; baseline must reach declared target')
                checked = verify_trained_artifacts(run,cfg,selected,release)
                state.update(status='completed',completed_epochs=len(commits),finished_utc=utc(),
                             last_resume=commits[-1]['checkpoint'],session_seconds=time.perf_counter()-started)
                write_json(run/'run.json',state)
                if mode=='smoke':
                    write_json(runs_root(root,cfg)/'gates/smoke.json',dict(passed=True,cuda=True,fingerprint=cfg['fingerprint'],
                        source_hashes=source_hashes(root),run=str(run.resolve()),
                        artifacts={str((run/'model_load.json').resolve()):file_hash(run/'model_load.json'),
                                   str((run/'checkpoints/best_macro_f1.pt').resolve()):file_hash(run/'checkpoints/best_macro_f1.pt')}))
                    size = (run/commits[-1]['checkpoint']).stat().st_size
                    mean_epoch = sum(h['epoch_seconds'] for h in read_json(run/'metrics/history.json'))/total_epochs
                    write_json(run/'baseline_estimate.json',dict(smoke_seconds_per_epoch=mean_epoch,
                        full_dataset_epoch_seconds_rough=mean_epoch*(9042/len([s for s in selected if s['split']=='train'])),
                        note='rough image-ratio estimate; full validation and storage overhead differ; measure baseline epoch1',
                        resume_checkpoint_bytes=size,baseline_50_epoch_checkpoint_bytes_estimate=size*54,
                        retention='every epoch kept; no old project artifacts deleted'))
                return run
            except ControlledStop as exc:
                commits,invalid = valid_commits(run)
                if invalid or not commits:
                    raise ValueError('Controlled stop has no safe epoch') from exc
                state.update(status='interrupted',reason=str(exc),completed_epochs=commits[-1]['epoch'],
                             last_resume=commits[-1]['checkpoint'],updated_utc=utc())
                write_json(run/'run.json',state)
                return run
            except BaseException as exc:
                oom = isinstance(exc,torch.cuda.OutOfMemoryError) or 'cuda out of memory' in str(exc).lower()
                state.update(status='oom' if oom else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                             error=f'{type(exc).__name__}: {exc}',updated_utc=utc())
                write_json(run/'run.json',state)
                (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                if not oom or resume_run or cfg['batch']==1:
                    raise
            finally:
                del trainer,resumed,context
                gc.collect()
                torch.cuda.empty_cache()
        if oom:
            attempts.append(dict(run=str(run),batch=cfg['batch'],reason='CUDA OOM; new pretrained initialization'))
            cfg = {**cfg,'batch':cfg['batch']//2}
            print(f"OOM attempt retained at {run}; NEW run from pretrained, batch={cfg['batch']}",flush=True)


def resume_probe(root: Path, cfg: dict) -> dict:
    """GPU: process mới tổng4 dừng1, process khác resume1→4 và đối chiếu."""
    if not environment()['cuda']:
        raise RuntimeError('Actual GPU resume probe not run: CUDA unavailable')
    require_gate(root,cfg,'smoke')
    token = uuid.uuid4().hex[:8]
    probe_cfg = root/'reports'/WORK/'verification'/f'probe_config_{token}.json'
    write_json(probe_cfg,cfg)
    result_file = probe_cfg.with_name('probe_result_'+token+'.json')
    script = root/'scripts/train_release.py'
    subprocess.run([sys.executable,str(script),'--config',str(probe_cfg),'train','--mode','resume_probe',
                    '--stop-after','1','--result-file',str(result_file)],cwd=root,check=True)
    run = Path(read_json(result_file)['run'])
    # OOM can create a new attempt with smaller batch: resume precisely its saved config.
    state = read_json(run/'run.json')
    ckpt = run/state['last_resume']
    subprocess.run([sys.executable,str(script),'--config',str(run/'config.json'),'resume','--run',str(run),
                    '--checkpoint',str(ckpt)],cwd=root,check=True)
    restored = read_json(run/'resume_restored_epoch_0001.json')
    history = read_json(run/'metrics/history.json')
    if (not restored['optimizer_verified'] or restored['next_epoch_one_based'] != 2
            or [h['epoch'] for h in history] != [1,2,3,4] or read_json(run/'run.json')['status'] != 'completed'):
        raise ValueError('Actual resume probe failed')
    evidence = dict(passed=True,cuda=True,at_utc=utc(),fingerprint=cfg['fingerprint'],source_hashes=source_hashes(root),
                    run=str(run.resolve()),separate_processes=True,epochs=[1,2,3,4],restored=restored,
                    artifacts={str((run/p).resolve()):file_hash(run/p) for p in ['resume_restored_epoch_0001.json','model_load.json','metrics/history.json']})
    write_json(runs_root(root,cfg)/'gates/resume.json',evidence)
    return evidence
