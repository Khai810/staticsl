"""Synthetic fixtures below verify software only; never experimental model results."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import socket
import os
import pytest
import torch
import yaml
from PIL import Image

from asl_yolo.core import digest, file_hash, read_json, write_json
from asl_yolo.protocol import CLASSES_29
from asl_yolo.train_release.config import load_training_config
from asl_yolo.train_release.dataset import audit_transforms, safe_path, select_smoke, verify_release
from asl_yolo.train_release.metrics import is_better, score_predictions, wilson
from asl_yolo.train_release.storage import (RunLock, commit_epoch, recover_run, valid_commits,
                                           verify_resume, rebuild_history, unlock_after_verification)


def sample(sid, label, class_id, split='val', background=False):
    return dict(sample_id=sid,label=label,class_id=class_id,split=split,
                boxes=[] if background else [[.5,.5,.8,.8]],
                capture_type='no_hand' if background else 'other_hand' if label=='nothing' else 'known_hand')


def test_metrics_rejections_support_zero_and_background():
    classes=['A','B','nothing']
    rows=[sample('a','A',0),sample('b','A',0),sample('c','B',1),sample('d','B',1),sample('bg','nothing',2,background=True)]
    predictions={'a':[dict(class_id=0,confidence=.9)],'b':[],
                 'c':[dict(class_id=2,confidence=.8)],'d':[dict(class_id=1,confidence=.7)],
                 'bg':[dict(class_id=2,confidence=.95)]}
    m=score_predictions(rows,predictions,classes,['A','B'],.25)
    assert m['accuracy']==.5 and m['n_positive']==4 and m['n_background']==1
    assert m['coverage']==.75 and m['accepted_accuracy']==pytest.approx(2/3)
    assert m['val_macro_f1_supported']==pytest.approx(2/3)
    assert m['per_class']['nothing']['f1'] is None and m['per_class']['nothing']['recall'] is None
    assert m['per_class']['nothing']['fp']==1 and m['per_class']['nothing']['background_fp']==1
    assert m['confusion_matrix'][0][-1]==1 and m['no_detection_positive']==1
    assert m['background_false_positive_rate']==1
    rejected=score_predictions(rows,predictions,classes,['A','B'],1.)
    assert rejected['accepted_accuracy'] is None and rejected['accuracy']==0
    assert rejected['per_class']['nothing']['f1'] is None
    assert rejected['background_correct_rejection_rate']==1
    with pytest.raises(ValueError,match='every sample'):
        score_predictions(rows,{},classes,['A','B'],.25)
    with pytest.raises(ValueError,match='Supported'):
        score_predictions(rows,predictions,classes,classes,.25)


def test_checkpoint_selection_and_wilson():
    a=dict(epoch=2,accuracy=.8,val_macro_f1_supported=.7)
    assert is_better(a,None)
    assert is_better({**a,'epoch':1},a)
    assert not is_better({**a,'epoch':3},a)
    assert is_better({**a,'accuracy':.81},a)
    assert is_better({**a,'accuracy':.1,'val_macro_f1_supported':.71},a)
    assert wilson(0,0) is None
    assert wilson(0,10)==pytest.approx([0,.2775327998628892])
    assert wilson(10,10)==pytest.approx([.7224672001371107,1])


@pytest.fixture
def synthetic_release(tmp_path):
    """Tiny synthetic images and weak-label schema; NOT ASL data."""
    release=tmp_path/'release'
    base=tmp_path/'base'
    base.mkdir()
    rows=[]
    for c in CLASSES_29:
        for split in ['train','val'] if c!='nothing' else ['train']:
            for j in range(22 if c=='nothing' else 1):
                rows.append(sample(f'{split}_{c}_{j}',c,CLASSES_29.index(c),split))
    rows.append(sample('train_background','nothing',27,'train',True))
    master=[]
    for i,row in enumerate(rows):
        row.update(image=f"draft/images/{row['split']}/{row['sample_id']}.png",
                   label_file=f"draft/labels/{row['split']}/{row['sample_id']}.txt")
        image,label=release/row['image'],release/row['label_file']
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(16,12),(i,2*i%255,3*i%255)).save(image)
        label.write_text('' if not row['boxes'] else f"{row['class_id']} 0.5 0.5 0.8 0.8\n")
        row.update(image_sha256=file_hash(image),label_sha256=file_hash(label))
        master.append(dict(row,role='development',exclusions=[],output_sha256=row['image_sha256'],
                           sha256=row['image_sha256'],pixel_sha256='pixel_'+str(i),split_group='group_'+str(i),session_id=None))
    for split in ['train','val']:
        (release/f'draft/{split}.txt').write_text('\n'.join('./'+str(Path(s['image']).relative_to('draft').as_posix()) for s in rows if s['split']==split)+'\n')
    (release/'draft/data.yaml').write_text(yaml.safe_dump(dict(names=dict(enumerate(CLASSES_29)))))
    counts={s:sum(r['split']==s for r in rows) for s in ['train','val']}
    meta=dict(kind='asl29_qc_run_release',samples=rows,classes=list(CLASSES_29),fingerprint=digest(rows),counts=counts,
              training_ready=True,evaluation_ready=False,run_split_frozen=True,review_kind='automated',human_review_claimed=False)
    write_json(release/'release.json',meta)
    (base/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in master))
    (base/'audit').mkdir()
    (base/'audit/phash_candidates.jsonl').write_text('')
    cfg=dict(release='release',base_dataset='base',fingerprint=meta['fingerprint'],expected_counts=counts,
             release_metadata_sha256={'release.json':file_hash(release/'release.json')},
             base_audit_sha256={n:file_hash(base/n) for n in ['manifest.jsonl','audit/phash_candidates.jsonl']})
    return tmp_path,cfg,rows,master


def test_adapter_29_mapping_empty_label_and_subset(synthetic_release):
    root,cfg,rows,_=synthetic_release
    checked=verify_release(root,cfg)
    assert checked['excluded_from_macro']==['nothing']
    assert checked['positive_support']['train']['nothing']==22
    assert checked['background_counts']=={'train':1}
    assert select_smoke(rows,42)==select_smoke(list(reversed(rows)),42)
    assert {s['sample_id']:s['split'] for s in select_smoke(rows,42)}.items() <= {s['sample_id']:s['split'] for s in rows}.items()
    with pytest.raises(ValueError,match='outside'):
        safe_path(root,'../escape')


def test_adapter_corrupt_label_rejected(synthetic_release):
    root,cfg,rows,_=synthetic_release
    (root/'release'/rows[0]['label_file']).write_text('27 0.5 0.5 0.8 0.8\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_release(root,cfg)


def test_unresolved_phash_blocks_without_declaring_duplicate(synthetic_release):
    root,cfg,rows,_=synthetic_release
    p=root/'base/audit/phash_candidates.jsonl'
    p.write_text(json.dumps(dict(a=rows[0]['sample_id'],b='holdout_id',cross_split=True))+'\n')
    cfg['base_audit_sha256']['audit/phash_candidates.jsonl']=file_hash(p)
    with pytest.raises(ValueError,match='pHash candidate'):
        verify_release(root,cfg)


def test_mapping_change_rejected_even_with_recomputed_release_hash(synthetic_release):
    root,cfg,_,_=synthetic_release
    p=root/'release/release.json'
    meta=read_json(p)
    meta['classes'][27],meta['classes'][28]=meta['classes'][28],meta['classes'][27]
    write_json(p,meta)
    cfg['release_metadata_sha256']['release.json']=file_hash(p)
    with pytest.raises(ValueError,match='mapping'):
        verify_release(root,cfg)


def synthetic_checkpoint(epoch, cfg):
    """Artificial torch module/optimizer for transaction tests only."""
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    model(torch.ones(1,2)).sum().backward()
    optimizer.step()
    return dict(epoch=epoch-1,ema=model,raw_model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler={'last_epoch':epoch-1},train_results={'epoch':list(range(1,epoch+1)),'loss':[.5]*epoch},
                release_context=dict(fingerprint=cfg['fingerprint'],config_sha256=digest(cfg),classes=['A','B']))


def make_run(tmp_path):
    cfg={'fingerprint':'SYNTHETIC_UNIT_TEST_ONLY'}
    run=tmp_path/'fixture_run'
    run.mkdir()
    write_json(run/'config.json',cfg)
    write_json(run/'class_mapping.json',['A','B'])
    write_json(run/'run.json',dict(status='interrupted',total_epochs=4))
    return run,cfg


def test_epoch_transaction_history_resume_and_recovery(tmp_path):
    run,cfg=make_run(tmp_path)
    for epoch in [1,2]:
        commit_epoch(run,synthetic_checkpoint(epoch,cfg),dict(epoch=epoch,accuracy=.5,val_macro_f1_supported=.5,
                     epoch_seconds=1.,native_fitness=.4),{'fixture_only':[]})
    commits,invalid=valid_commits(run)
    assert not invalid and [c['epoch'] for c in commits]==[1,2]
    assert [h['epoch'] for h in read_json(run/'metrics/history.json')]==[1,2]
    restored=verify_resume(run,run/'checkpoints/last_resume.pt',cfg)
    assert restored['epoch']==1 and restored['optimizer']['state']
    with pytest.raises(ValueError,match='latest valid'):
        verify_resume(run,run/commits[0]['checkpoint'],cfg)
    with pytest.raises(ValueError,match='duplicate'):
        commit_epoch(run,synthetic_checkpoint(2,cfg),dict(epoch=2),{})
    # Simulate dying while writing generation 3 before its commit marker.
    p=run/'checkpoints/unfinished.pt.partial'
    p.write_bytes(b'partial checkpoint fixture')
    result=recover_run(run)
    assert result['recovered_epoch']==2 and p.exists()
    assert read_json(run/'checkpoints/best_macro_f1.json')['selection']['epoch']==1
    assert read_json(run/'metrics/history.json')[1]['epoch']==2
    state=read_json(run/'run.json'); state['status']='completed'; write_json(run/'run.json',state)
    with pytest.raises(ValueError,match='Completed'):
        verify_resume(run,run/'checkpoints/last_resume.pt',cfg)


def test_corrupted_latest_generation_falls_back_only_via_recover(tmp_path):
    run,cfg=make_run(tmp_path)
    records=[]
    for epoch in [1,2]:
        records.append(commit_epoch(run,synthetic_checkpoint(epoch,cfg),dict(epoch=epoch,accuracy=.5,
            val_macro_f1_supported=.5,native_fitness=.1,epoch_seconds=1.),{}))
    (run/records[1]['checkpoint']).write_bytes(b'SYNTHETIC CORRUPTION')
    with pytest.raises(ValueError,match='safe complete'):
        verify_resume(run,run/'checkpoints/last_resume.pt',cfg)
    result=recover_run(run)
    assert result['recovered_epoch']==1
    assert len(valid_commits(run)[0])==1
    assert len(list((run/'recovery').rglob('epoch_0002.json')))==1


def test_lock_prevents_live_writer_and_requires_remote_attestation(tmp_path):
    run,_=make_run(tmp_path)
    with RunLock(run):
        with pytest.raises(RuntimeError,match='still exists'):
            with RunLock(run): pass
        with pytest.raises(RuntimeError,match='still alive'):
            unlock_after_verification(run,True)
    write_json(run/'.writer.lock',dict(host='OTHER_RUNTIME_SYNTHETIC',pid=123,token='fixture'))
    with pytest.raises(RuntimeError,match='another host'):
        with RunLock(run): pass
    with pytest.raises(RuntimeError,match='explicit'):
        unlock_after_verification(run,False)
    unlock_after_verification(run,True)
    assert not (run/'.writer.lock').exists()


def test_production_config_never_unlocks_final_test():
    root=Path(__file__).resolve().parents[1]
    cfg=load_training_config(root/'configs/asl_yolo11n_auto_v2_training.json')
    assert cfg['require_cuda'] and not cfg['allow_final_test']


def test_implicit_augmentation_and_corrupt_lock_are_detected(tmp_path):
    class Albumentations:
        transform=type('MockCompose',(),{'transforms':['unexpected fixture transform']})()
    with pytest.raises(ValueError,match='implicit'):
        audit_transforms(Albumentations(),{},True)
    run,_=make_run(tmp_path)
    (run/'.writer.lock').write_text('{unfinished')
    with pytest.raises(RuntimeError,match='Incomplete lock'):
        with RunLock(run): pass
    unlock_after_verification(run,True)
    assert not (run/'.writer.lock').exists()


def test_final_eval_extra_callback_has_no_epoch_commit():
    """Structural check against installed source; GPU behavior still requires real probe."""
    import inspect
    from asl_yolo.train_release.engine import trainer_class
    cls=trainer_class({'run':'.','cfg':{}})
    source=inspect.getsource(cls.run_callbacks)
    assert 'on_fit_epoch_end' not in source and 'commit_epoch' not in source


def test_validator_keeps_top_two_and_scales_boxes(monkeypatch):
    """Synthetic post-NMS tensors only; no model/dataset quality metrics."""
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from asl_yolo.train_release.engine import validator_class
    monkeypatch.setattr(DetectionValidator,'update_metrics',lambda *args:None)
    cls=validator_class()
    validator=cls.__new__(cls)
    validator.image_predictions={}
    validator._prepare_batch=lambda i,b:dict(imgsz=(320,320),ori_shape=(200,200),ratio_pad=((1.6,1.6),(0,0)))
    validator._prepare_pred=lambda p:p
    pred=dict(cls=torch.tensor([0.,1.,2.]),conf=torch.tensor([.1,.9,.8]),
              bboxes=torch.tensor([[0.,0.,160.,160.]]*3))
    batch={'im_file':['synthetic_sample.png']}
    validator.update_metrics([pred],batch)
    kept=validator.image_predictions['synthetic_sample']
    assert [p['class_id'] for p in kept]==[1,2]
    assert kept[0]['box']==pytest.approx([0,0,100,100])
    with pytest.raises(ValueError,match='Repeated'):
        validator.update_metrics([pred],batch)
