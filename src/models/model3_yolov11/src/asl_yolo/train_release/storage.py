"""Journal từng epoch; commit marker và checksum là nguồn phục hồi."""
from __future__ import annotations
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import socket
import uuid

from ..core import digest, file_hash, read_json, write_json
from .metrics import is_better


class ControlledStop(RuntimeError):
    """Ngắt thử nghiệm ngay sau một epoch đã commit bền vững."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish_bytes(path: Path, data: bytes) -> str:
    """Ghi tạm, fsync và đọc kiểm tra trước công bố; không hứa Drive atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep temporary names short enough for the project's deep Windows path.
    partial = path.with_name('.p_' + uuid.uuid4().hex[:8] + '.partial')
    import hashlib
    expected = hashlib.sha256(data).hexdigest()
    with partial.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if file_hash(partial) != expected:
        raise IOError('Storage verification failed before publish')
    os.replace(partial, path)
    if file_hash(path) != expected:
        raise IOError('Storage verification failed after publish')
    return expected


def publish_copy(source: Path, target: Path) -> str:
    return publish_bytes(target, source.read_bytes())


def publish_json(path: Path, value) -> str:
    return publish_bytes(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))


class RunLock(AbstractContextManager):
    """Sentinel single writer; host khác phải xác minh job cũ đã dừng."""
    def __init__(self, run: Path):
        self.path = run / '.writer.lock'
        self.owner = dict(host=socket.gethostname(), pid=os.getpid(), started_utc=utc(), token=uuid.uuid4().hex)
        import psutil
        self.owner['process_created'] = psutil.Process().create_time()

    def __enter__(self):
        if self.path.exists():
            try:
                owner = read_json(self.path)
            except (ValueError,OSError) as exc:
                raise RuntimeError('Incomplete lock: verify old job stopped, then unlock --old-job-stopped') from exc
            if owner['host'] != socket.gethostname():
                raise RuntimeError('Lock from another host: verify old GPU job stopped, then use unlock --old-job-stopped')
            import psutil
            if psutil.pid_exists(owner['pid']) and (not owner.get('process_created') or
                    psutil.Process(owner['pid']).create_time()==owner['process_created']):
                raise RuntimeError(f"Writer process {owner['pid']} still exists; do not resume")
            self.path.rename(self.path.with_name('.writer.stale.' + uuid.uuid4().hex + '.json'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x', encoding='utf-8') as stream:
            json.dump(self.owner, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if read_json(self.path)['token'] != self.owner['token']:
            raise RuntimeError('Concurrent writer detected')
        return self

    def __exit__(self, *exc):
        if self.path.exists() and read_json(self.path).get('token') == self.owner['token']:
            self.path.unlink()
        return False


def unlock_after_verification(run: Path, old_job_stopped: bool) -> None:
    """Chỉ gỡ lock khác host khi người vận hành xác nhận job cũ đã dừng."""
    path = run / '.writer.lock'
    if not old_job_stopped:
        raise RuntimeError('Missing explicit --old-job-stopped verification')
    try:
        owner = read_json(path)
    except ValueError:
        owner = dict(host='unknown_partial_lock',raw_contents=path.read_text(encoding='utf-8',errors='replace'))
    import psutil
    if owner['host'] == socket.gethostname() and psutil.pid_exists(owner['pid']):
        raise RuntimeError('Local writer still alive')
    write_json(run / ('unlock_' + uuid.uuid4().hex + '.json'), dict(at_utc=utc(), previous=owner,
               operator_attestation='old GPU process stopped; browser disconnect alone is insufficient'))
    path.rename(run / ('.writer.released.' + uuid.uuid4().hex + '.json'))


def valid_commits(run: Path) -> tuple[list[dict], list[str]]:
    """Đọc journal, bỏ generation thiếu/corrupt; báo rõ trong status."""
    commits, invalid = [], []
    for marker in sorted((run / 'commits').glob('epoch_*.json')):
        try:
            record = read_json(marker)
            for relative, sha in record['files'].items():
                path = (run / relative).resolve()
                if not path.is_relative_to(run.resolve()) or file_hash(path) != sha:
                    raise ValueError(f'Bad committed artifact: {relative}')
            commits.append(record)
        except (OSError, ValueError, KeyError) as exc:
            invalid.append(f'{marker.name}: {exc}')
    return commits, invalid


def rebuild_history(run: Path, commits: list[dict]) -> None:
    """Tái tạo history từ epoch đã commit, không append trùng sau resume."""
    epochs = [c['epoch'] for c in commits]
    if epochs != list(range(1, len(epochs)+1)):
        raise ValueError('Journal epochs must be contiguous and unique')
    history = [{k:v for k,v in read_json(run / c['metrics']).items()
                if isinstance(v,(int,float,str)) or v is None} for c in commits]
    publish_json(run / 'metrics/history.json', history)
    flat = [{k:v for k,v in h.items() if isinstance(v, (int,float,str)) or v is None} for h in history]
    if flat:
        stream = io.StringIO(newline='')
        fields = list(dict.fromkeys(k for row in flat for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
        publish_bytes(run / 'metrics/history.csv', stream.getvalue().encode('utf-8'))


def commit_epoch(run: Path, checkpoint: dict, metrics: dict, predictions: dict) -> dict:
    """Ghi immutable checkpoint/metrics/predictions rồi commit marker cuối."""
    import torch
    commits, invalid = valid_commits(run)
    if invalid:
        raise ValueError(f'Invalid journal must be inspected: {invalid}')
    epoch = metrics['epoch']
    if epoch != len(commits)+1 or checkpoint['epoch']+1 != epoch:
        raise ValueError('Refuse duplicate/out-of-order epoch commit')
    generation = f'checkpoints/epoch_{epoch:04d}_{uuid.uuid4().hex[:8]}'
    target = run / generation / 'resume.pt'
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    sha = publish_bytes(target, buffer.getvalue())
    loaded = torch.load(target, map_location='cpu', weights_only=False)
    if loaded['epoch']+1 != epoch or loaded.get('optimizer') is None or loaded.get('raw_model') is None:
        raise ValueError('Saved checkpoint lacks optimizer/raw model/epoch')
    metric_path, pred_path = generation+'/metrics.json', generation+'/predictions.json'
    publish_json(run / metric_path, metrics)
    publish_json(run / pred_path, predictions)
    files = {generation+'/resume.pt':sha, metric_path:file_hash(run / metric_path), pred_path:file_hash(run / pred_path)}
    record = dict(epoch=epoch, checkpoint=generation+'/resume.pt', metrics=metric_path, predictions=pred_path,
                  files=files, config_sha256=checkpoint['release_context']['config_sha256'], committed_utc=utc())
    publish_json(run / f'commits/epoch_{epoch:04d}.json', record)
    # Aliases are convenience copies; immutable journal generation remains the fallback.
    publish_copy(target, run / 'checkpoints/last_resume.pt')
    publish_json(run / 'checkpoints/last_resume.json', record)
    previous = read_json(run / 'checkpoints/best_macro_f1.json') if (run / 'checkpoints/best_macro_f1.json').exists() else None
    if is_better(metrics, previous['selection'] if previous else None):
        inference = {k:v for k,v in checkpoint.items() if k not in ('raw_model','optimizer','scheduler','rng','scaler')}
        inference.update(epoch=-1, optimizer=None)
        buffer = io.BytesIO()
        torch.save(inference, buffer)
        best_sha = publish_bytes(run / 'checkpoints/best_macro_f1.pt', buffer.getvalue())
        publish_json(run / 'checkpoints/best_macro_f1.json', dict(selection={k:metrics[k] for k in ('epoch','accuracy','val_macro_f1_supported')},
                   checkpoint_sha256=best_sha, generation=record, role='EMA float32 inference; not resume'))
    rebuild_history(run, commits+[record])
    publish_json(run / 'progress.json', dict(epoch=epoch, latest=record, at_utc=utc()))
    return record


def verify_resume(run: Path, checkpoint: Path, cfg: dict) -> dict:
    """Chỉ resume checkpoint mới nhất đã commit trong run chưa hoàn tất."""
    import torch
    state = read_json(run / 'run.json')
    if state['status'] == 'completed':
        raise ValueError('Completed run cannot resume; warm start/extension requires a new experiment')
    if read_json(run / 'config.json') != cfg:
        raise ValueError('Resume config changed')
    commits, invalid = valid_commits(run)
    if invalid or not commits:
        raise ValueError(f'No safe complete journal: {invalid}')
    rebuild_history(run, commits)
    latest = commits[-1]
    expected = latest['files'][latest['checkpoint']]
    if file_hash(checkpoint) != expected:
        raise ValueError('Select the latest valid committed checkpoint explicitly; older/incomplete alias rejected')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    context = ckpt['release_context']
    if (context['fingerprint'] != cfg['fingerprint'] or context['config_sha256'] != digest(cfg)
            or context['classes'] != read_json(run / 'class_mapping.json')
            or ckpt['epoch']+1 != latest['epoch'] or latest['epoch'] >= state['total_epochs']
            or ckpt.get('optimizer') is None or ckpt.get('raw_model') is None or ckpt.get('scheduler') is None):
        raise ValueError('Checkpoint context/state does not support continuation')
    repair_derived(run,commits)
    return ckpt


def repair_derived(run: Path, commits: list[dict]) -> None:
    """Khôi phục aliases và native CSV từ journal, kể cả ngắt sau commit marker."""
    import torch
    rebuild_history(run,commits)
    rows = [read_json(run/c['metrics']) for c in commits]
    chosen = None
    for record,row in zip(commits,rows):
        if chosen is None or is_better(row,chosen[1]):
            chosen = (record,row)
    checkpoint = torch.load(run/chosen[0]['checkpoint'],map_location='cpu',weights_only=False)
    inference = {k:v for k,v in checkpoint.items() if k not in ('raw_model','optimizer','scheduler','rng','scaler')}
    inference.update(epoch=-1,optimizer=None)
    buffer = io.BytesIO()
    torch.save(inference,buffer)
    sha = publish_bytes(run/'checkpoints/best_macro_f1.pt',buffer.getvalue())
    write_json(run/'checkpoints/best_macro_f1.json',dict(selection={k:chosen[1][k] for k in ('epoch','accuracy','val_macro_f1_supported')},
               checkpoint_sha256=sha,generation=chosen[0],role='EMA float32 inference; not resume'))
    publish_copy(run/commits[-1]['checkpoint'],run/'checkpoints/last_resume.pt')
    write_json(run/'checkpoints/last_resume.json',commits[-1])
    latest = torch.load(run/commits[-1]['checkpoint'],map_location='cpu',weights_only=False)
    results = latest.get('train_results',{})
    if results:
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))
        path = run/'framework/results.csv'
        if path.exists():
            publish_copy(path,run/('recovery/csv_'+uuid.uuid4().hex[:8]+'.csv'))
        publish_bytes(path,stream.getvalue().encode('utf-8'))
    native_index = max(range(len(rows)),key=lambda i:(rows[i]['native_fitness'],rows[i]['epoch']))
    publish_copy(run/commits[native_index]['checkpoint'],run/'framework/weights/best.pt')
    publish_copy(run/commits[-1]['checkpoint'],run/'framework/weights/last.pt')


def recover_run(run: Path) -> dict:
    """Lệnh phục hồi rõ ràng về prefix epoch hợp lệ; giữ file hỏng để kiểm tra."""
    with RunLock(run):
        state = read_json(run/'run.json')
        if state['status']=='completed':
            raise ValueError('Completed run is immutable; inspect artifacts instead')
        valid,invalid = valid_commits(run)
        prefix = []
        for record in valid:
            if record['epoch'] != len(prefix)+1:
                break
            prefix.append(record)
        if not prefix:
            raise ValueError('No completed epoch can be recovered; start a new run from pretrained')
        recovery = run/'recovery'/uuid.uuid4().hex
        recovery.mkdir(parents=True)
        keep = {f"epoch_{c['epoch']:04d}.json" for c in prefix}
        for marker in (run/'commits').glob('epoch_*.json'):
            if marker.name not in keep:
                marker.rename(recovery/marker.name)
        repair_derived(run,prefix)
        result = dict(at_utc=utc(),recovered_epoch=prefix[-1]['epoch'],checkpoint=prefix[-1]['checkpoint'],
                      retained_invalid_artifacts=invalid,partial_files=[str(p.relative_to(run)) for p in run.rglob('*.partial')],
                      policy='No checkpoint files deleted; choose returned immutable checkpoint explicitly')
        write_json(recovery/'recovery.json',result)
        state.update(status='interrupted',completed_epochs=prefix[-1]['epoch'],last_resume=prefix[-1]['checkpoint'],updated_utc=utc())
        write_json(run/'run.json',state)
        return result
