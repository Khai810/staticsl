"""Chỉ validation: dữ liệu nguồn, sweep, hình 300 dpi và báo cáo truy vết."""
from __future__ import annotations
from collections import Counter
import csv
import html
from pathlib import Path
import json
import numpy as np
from ..core import digest, file_hash, read_json, read_rgb, write_json
from ..training import assert_mapping
from .config import source_hashes
from .dataset import prepare_dataset, verify_release
from .engine import validator_class
from .metrics import score_predictions, top_detection
from .storage import RunLock, utc, valid_commits


def write_csv(path: Path, rows: list[dict]) -> None:
    """CSV rỗng chỉ có trạng thái trong JSON, không bịa dòng số liệu."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    return plt


def save_figure(fig, path: Path) -> None:
    """PNG 300 dpi và SVG từ cùng dữ liệu, không phụ thuộc notebook."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    pyplot().close(fig)


def plot_distribution(samples: list[dict], classes: list[str], output: Path) -> None:
    """Phân bố thật train/val, positive nothing tách khỏi background."""
    rows=[]
    for c in classes:
        rows.append(dict(label=c,train_positive=sum(s['split']=='train' and s['label']==c and bool(s['boxes']) for s in samples),
                         val_positive=sum(s['split']=='val' and s['label']==c and bool(s['boxes']) for s in samples)))
    rows.append(dict(label='BACKGROUND',train_positive=sum(s['split']=='train' and not s['boxes'] for s in samples),
                     val_positive=sum(s['split']=='val' and not s['boxes'] for s in samples)))
    write_csv(output/'distribution.csv',rows)
    write_json(output/'distribution.json',rows)
    plt=pyplot()
    fig,ax=plt.subplots(figsize=(14,5))
    x=np.arange(len(rows))
    ax.bar(x-.2,[r['train_positive'] for r in rows],width=.4,label='Train')
    ax.bar(x+.2,[r['val_positive'] for r in rows],width=.4,label='Validation')
    ax.set(xticks=x,xticklabels=[r['label'] for r in rows],ylabel='Images (count)',
           title='Frozen auto V2 release: positive labels and empty-label backgrounds')
    ax.tick_params(axis='x',rotation=90)
    ax.legend()
    save_figure(fig,output/'distribution')


def artifact_manifest(run: Path) -> None:
    """Phụ lục checksum artifact; bỏ lock, partial và chính manifest."""
    files={p.relative_to(run).as_posix():file_hash(p) for p in run.rglob('*') if p.is_file()
           and p.name!='artifact_checksums.json' and not p.name.startswith('.writer') and not p.name.endswith('.partial')}
    write_json(run/'artifact_checksums.json',dict(at_utc=utc(),files=files))


def export_gallery(root: Path, run: Path, samples: list[dict], predictions: dict, score: dict, cfg: dict) -> None:
    """GT tự động xanh, prediction đỏ; chỉ ảnh val đã được khóa."""
    from PIL import Image, ImageDraw
    classes=read_json(run/'class_mapping.json')
    by_id={s['sample_id']:s for s in samples}
    groups={'correct':[],'wrong_class':[],'no_detection':[],'background_fp':[],'high_confidence_wrong':[]}
    for row in score['records']:
        if row['background']:
            if not row['correct']: groups['background_fp'].append(row)
        elif row['correct']:
            groups['correct'].append(row)
        elif row['predicted']=='NO_DETECTION':
            groups['no_detection'].append(row)
        else:
            groups['wrong_class'].append(row)
            if row['confidence'] >= .8: groups['high_confidence_wrong'].append(row)
    records=[]
    output=run/'figures/gallery'
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in groups.items():
        rows=sorted(rows,key=lambda r: (-(r['confidence'] or 0),r['sample_id']))[:12]
        for row in rows:
            sample=by_id[row['sample_id']]
            image=Image.fromarray(read_rgb(root/cfg['release']/sample['image']))
            image.thumbnail((700,500))
            w,h=image.size
            canvas=Image.new('RGB',(max(w,700),h+85),'white')
            canvas.paste(image,(0,0))
            draw=ImageDraw.Draw(canvas)
            for x,y,bw,bh in sample['boxes']:
                draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='lime',width=3)
            original=read_rgb(root/cfg['release']/sample['image'])
            oh,ow=original.shape[:2]
            top=top_detection(predictions[row['sample_id']],score['threshold'])
            if top:
                x1,y1,x2,y2=top['box']
                draw.rectangle((x1*w/ow,y1*h/oh,x2*w/ow,y2*h/oh),outline='red',width=2)
            text=(f"val | {row['sample_id']} | {name}\nGT auto: {row['truth']} | Pred: {row['predicted']} | conf: {row['confidence']}\n"
                  'Green: automated GT box; Red: predicted box; weak labels')
            draw.multiline_text((5,h+5),text,fill='black',spacing=4)
            filename=name+'_'+row['sample_id']+'.png'
            canvas.save(output/filename,dpi=(300,300))
            records.append(dict(**row,group=name,file=filename,reason=name,gt_box_source='automated weak localization'))
    write_json(output/'index.json',dict(groups_total={k:len(v) for k,v in groups.items()},examples=records))
    cards='\n'.join(f"<figure><img src='{html.escape(r['file'])}' width='700'><figcaption>{html.escape(r['group'])}: {html.escape(r['sample_id'])}</figcaption></figure>" for r in records)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Validation error gallery</title><h1>Validation examples</h1><p>Automated GT, no expert ASL validation. Missing groups mean no such examples observed.</p>'+cards,encoding='utf-8')


def analyze_validation(root: Path, run: Path, cfg: dict) -> dict:
    """Khóa model đã train, sweep confidence chỉ full validation."""
    if cfg['allow_final_test']:
        raise ValueError('Final test must remain disabled')
    with RunLock(run):
        state=read_json(run/'run.json')
        if state['status']!='completed' or state['mode']!='baseline':
            raise ValueError('Full validation analysis requires a completed baseline, not smoke/random weights')
        if read_json(run/'config.json')!=cfg or read_json(run/'source_hashes.json')!=source_hashes(root):
            raise ValueError('Run config/source mismatch')
        checked=verify_release(root,cfg)
        commits,invalid=valid_commits(run)
        if invalid or len(commits)!=state['total_epochs']:
            raise ValueError('Baseline journal incomplete')
        best=run/'checkpoints/best_macro_f1.pt'
        best_meta=read_json(run/'checkpoints/best_macro_f1.json')
        sha=file_hash(best)
        if sha!=best_meta['checkpoint_sha256']:
            raise ValueError('Selected checkpoint checksum changed')
        if (run/'inference.json').exists():
            locked=read_json(run/'inference.json')
            if locked['checkpoint_sha256']!=sha or locked['fingerprint']!=cfg['fingerprint']:
                raise ValueError('Immutable inference lock does not match checkpoint/release')
            for relative,expected in locked.get('analysis_sources',{}).items():
                if file_hash(run/relative)!=expected:
                    raise ValueError('Locked validation source corrupted; inspect artifacts')
            finalize_analysis(root,run,cfg,checked)
            return locked
        selection=dict(at_utc=utc(),checkpoint_sha256=sha,epoch=best_meta['selection']['epoch'],
                       criterion='val_macro_f1_supported at .25; then Accuracy; then earlier epoch',
                       supported_classes=checked['supported_classes'],fingerprint=cfg['fingerprint'])
        if (run/'selection_lock.json').exists() and read_json(run/'selection_lock.json')['checkpoint_sha256']!=sha:
            raise ValueError('Checkpoint was already locked before sweep')
        write_json(run/'selection_lock.json',selection)
        data_yaml=prepare_dataset(root/cfg['release'],checked['samples'],root/'.cache/train_release'/cfg['fingerprint'][:12]/'full')
        args=dict(data=str(data_yaml),imgsz=cfg['imgsz'],batch=cfg['batch'],device=0,workers=0,conf=.001,
                  iou=cfg['nms_iou'],max_det=300,augment=False,plots=True,split='val',task='detect',
                  half=False,save_json=False,save_txt=False)
        validator=validator_class()(args=args,save_dir=run/'figures/detection_native')
        native=validator(model=str(best))
        assert_mapping(validator.names,checked['metadata']['classes'])
        samples=[s for s in checked['samples'] if s['split']=='val']
        predictions=validator.image_predictions
        write_json(run/'predictions/validation.json',dict(checkpoint_sha256=sha,fingerprint=cfg['fingerprint'],
                   split='val',storage='top 2 after native class-aware multi_label=True NMS; sufficient for all highest-box thresholds',
                   predictions=predictions))
        curves=[]
        for x,y,xlabel,ylabel in validator.metrics.curves_results:
            curves.append(dict(x=np.asarray(x).tolist(),y=np.asarray(y).tolist(),xlabel=xlabel,ylabel=ylabel))
        write_json(run/'metrics/detection_curves.json',curves)
        write_json(run/'metrics/detection_validation.json',dict(checkpoint_sha256=sha,split='val',metrics=native,
                   label_quality='automated weak localization labels; not expert localization evaluation'))
        scores=[score_predictions(samples,predictions,checked['metadata']['classes'],checked['supported_classes'],t) for t in cfg['sweep_grid']]
        winner=max(scores,key=lambda s:(s['val_macro_f1_supported'],s['accuracy'],s['threshold']))
        write_json(run/'metrics/validation_sweep.json',[{k:v for k,v in s.items() if k!='records'} for s in scores])
        write_csv(run/'metrics/validation_sweep.csv',[{k:v for k,v in s.items() if isinstance(v,(int,float)) or v is None} for s in scores])
        fixed=score_predictions(samples,predictions,checked['metadata']['classes'],checked['supported_classes'],cfg['monitor_confidence'])
        write_json(run/'metrics/validation_fixed.json',fixed)
        write_json(run/'metrics/validation_selected.json',winner)
        write_csv(run/'predictions/validation_selected.csv',winner['records'])
        write_csv(run/'metrics/per_class.csv',[dict(label=c,**v) for c,v in winner['per_class'].items()])
        inference=dict(checkpoint='checkpoints/best_macro_f1.pt',checkpoint_sha256=sha,classes=checked['metadata']['classes'],
                       fingerprint=cfg['fingerprint'],confidence=winner['threshold'],imgsz=cfg['imgsz'],nms_iou=cfg['nms_iou'],
                       max_det=300,multi_label=True,agnostic_nms=False,rule='highest confidence after class-aware NMS; reject=NO_DETECTION',
                       selected_on='val',selection='macro F1 supported, then Accuracy, then higher threshold',
                       grid=cfg['sweep_grid'],macro_class_count=len(checked['supported_classes']),
                       excluded_from_macro=checked['excluded_from_macro'],allow_final_test=False,
                       development_result=True,at_utc=utc(),analysis_sources={p:file_hash(run/p) for p in
                       ['predictions/validation.json','metrics/validation_selected.json','metrics/validation_fixed.json',
                        'metrics/validation_sweep.json','metrics/detection_curves.json','metrics/detection_validation.json']})
        write_json(run/'inference.json',inference)
        finalize_analysis(root,run,cfg,checked)
        return inference


def finalize_analysis(root: Path, run: Path, cfg: dict, checked: dict) -> None:
    """Chạy lại được nếu phiên chết sau khóa threshold, không tune lần nữa."""
    predictions=read_json(run/'predictions/validation.json')['predictions']
    winner=read_json(run/'metrics/validation_selected.json')
    samples=[s for s in checked['samples'] if s['split']=='val']
    plot_distribution(checked['samples'],checked['metadata']['classes'],run/'figures/data')
    export_gallery(root,run,samples,predictions,winner,cfg)
    write_csv(run/'metrics/hyperparameters.csv',[dict(parameter=k,value=v) for k,v in read_json(run/'resolved_config.json').items()])
    regenerate_figures(run)
    write_validation_report(run)
    artifact_manifest(run)


def regenerate_figures(run: Path) -> None:
    """Vẽ lại hoàn toàn từ CSV/JSON đã lưu, không inference/test."""
    lock=read_json(run/'inference.json')
    for relative,expected in lock.get('analysis_sources',{}).items():
        if file_hash(run/relative)!=expected:
            raise ValueError('Locked validation source corrupted')
    history=read_json(run/'metrics/history.json')
    selected=read_json(run/'metrics/validation_selected.json')
    sweep=read_json(run/'metrics/validation_sweep.json')
    plt=pyplot()
    epochs=[h['epoch'] for h in history]
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ax,loss in zip(axes[0],['box_loss','cls_loss','dfl_loss']):
        for split in ('train','val'):
            key=f'{split}/{loss}'
            if any(key in h for h in history):
                ax.plot(epochs,[h.get(key,np.nan) for h in history],label=split)
        ax.set(title=loss,xlabel='Completed epoch',ylabel='Loss (framework units)')
        ax.legend()
    for key in ('accuracy','val_macro_f1_supported'):
        axes[1,0].plot(epochs,[h[key] for h in history],label=key)
    axes[1,0].set(title='Validation at confidence 0.25',xlabel='Completed epoch',ylabel='Score (0–1)')
    axes[1,0].legend()
    for key in history[0]:
        if key.startswith('lr/'):
            axes[1,1].plot(epochs,[h[key] for h in history],label=key)
    axes[1,1].set(xlabel='Completed epoch',ylabel='Learning rate',title='Actual optimizer learning rate')
    axes[1,1].legend()
    axes[1,2].plot(epochs,[h['epoch_seconds'] for h in history],label='Train + val + native save')
    axes[1,2].set(xlabel='Completed epoch',ylabel='Seconds',title='Epoch runtime (journal commit excluded)')
    save_figure(fig,run/'figures/learning_curves')
    fig,ax=plt.subplots(figsize=(10,5))
    for key in ['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']:
        if key in history[0]:
            ax.plot(epochs,[h[key] for h in history],label=key)
    ax.set(xlabel='Completed epoch',ylabel='Detection score (0–1)',title='Validation detection on automated weak boxes')
    ax.legend()
    save_figure(fig,run/'figures/detection_history')
    cm=np.asarray(selected['confusion_matrix'],dtype=float)
    for normalized in (False,True):
        matrix=np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(axis=1,keepdims=True)!=0) if normalized else cm
        write_csv(run/f"metrics/confusion_{'normalized' if normalized else 'counts'}.csv",[dict(truth=c,**dict(zip(selected['confusion_columns'],row))) for c,row in zip(selected['confusion_rows'],matrix.tolist())])
        fig,ax=plt.subplots(figsize=(15,12))
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if normalized else None)
        ax.set(xticks=range(cm.shape[1]),xticklabels=selected['confusion_columns'],yticks=range(cm.shape[0]),
               yticklabels=selected['confusion_rows'],xlabel='Predicted (includes NO_DETECTION)',ylabel='GT source label',
               title='Validation confusion '+('row-normalized; zero-support nothing row is N/A' if normalized else 'counts'))
        ax.tick_params(axis='x',rotation=90)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if matrix[i,j]: ax.text(j,i,f'{matrix[i,j]:.2f}' if normalized else str(int(matrix[i,j])),ha='center',va='center',fontsize=6)
        fig.colorbar(im,ax=ax,label='Fraction of class' if normalized else 'Images')
        save_figure(fig,run/f"figures/confusion_{'normalized' if normalized else 'counts'}")
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
    classes=selected['confusion_rows']
    axes[0].bar(classes,[selected['per_class'][c]['f1'] if selected['per_class'][c]['f1'] is not None else np.nan for c in classes])
    axes[0].set(ylabel='F1 (0–1)',ylim=(0,1),title='Validation per class; nothing F1=N/A (support=0)')
    axes[1].bar(classes,[selected['per_class'][c]['support'] for c in classes])
    axes[1].set(ylabel='Positive support (images)',xlabel='GT source label')
    save_figure(fig,run/'figures/per_class')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,label in zip(axes,['coverage','accepted_accuracy','accuracy'],['Coverage','Accuracy among accepted positives','Accuracy among all positives']):
        ax.plot([s['threshold'] for s in sweep],[s[key] if s[key] is not None else np.nan for s in sweep],marker='o')
        ax.axvline(selected['threshold'],linestyle='--',color='gray')
        ax.set(xlabel='Confidence threshold',ylabel='Fraction (0–1)',ylim=(0,1.02),title=label+' — validation')
    save_figure(fig,run/'figures/coverage_threshold')
    for i,curve in enumerate(read_json(run/'metrics/detection_curves.json')):
        y=np.asarray(curve['y'])
        if y.ndim==1: y=y[None,:]
        fig,ax=plt.subplots(figsize=(8,5))
        for row in y: ax.plot(curve['x'],row,alpha=.15,color='steelblue')
        if len(y): ax.plot(curve['x'],y.mean(axis=0),label='Mean across supported detection classes',color='navy')
        ax.set(xlabel=curve['xlabel'],ylabel=curve['ylabel'],title='Validation detection curve — weak localization labels')
        ax.legend()
        save_figure(fig,run/f'figures/detection_curve_{i}')
    lock=read_json(run/'inference.json')
    caption=(f"Validation: {selected['n_positive']} positive images and {selected['n_background']} empty-label backgrounds. "
             f"YOLO11n, selected epoch {read_json(run/'selection_lock.json')['epoch']}, threshold {lock['confidence']:.2f}. "
             f"Macro F1 over {selected['macro_class_count']} supported classes; nothing has zero positive validation support. "
             'Boxes are automated weak labels. Checkpoint and threshold were chosen on validation; these are development results. '
             'Frame correlation weakens independence assumptions; these results do not establish performance on new signers.')
    write_json(run/'figures/captions.json',{p.stem:caption for p in (run/'figures').glob('*.png')})


def write_validation_report(run: Path) -> None:
    m=read_json(run/'metrics/validation_selected.json')
    fixed=read_json(run/'metrics/validation_fixed.json')
    cfg=read_json(run/'config.json')
    state=read_json(run/'run.json')
    detection=read_json(run/'metrics/detection_validation.json')['metrics']
    selected=read_json(run/'checkpoints/best_macro_f1.json')['selection']
    interval=m['accuracy_wilson95']
    text=f'''# Kết quả phát triển YOLO11n trên validation auto V2

Đã train thật {state['completed_epochs']} epoch trên CUDA. Dữ liệu: 9.042 train / 3.909 validation; mapping 29 đầu ra, 28 lớp có positive validation. Review tự động được người dùng cho phép, không phải annotation chuyên gia ASL. Split nguồn 5/3/2 là đề xuất, subset train/val đã khóa; session phần lớn chưa biết. Không chạy final test hoặc external.

## Phương pháp và cấu hình

YOLO11n pretrained COCO mới, ảnh RGB giữ tỷ lệ và letterbox, imgsz=320, batch={cfg['batch']}, seed=42, freeze=0, AMP=False. SGD, lr0=.01, momentum=.937, weight_decay=.0005, linear decay lrf=.01, warmup3; augmentation train xoay ±12° và scale .07, mọi biến thể khác theo YAML khóa; val không augmentation. Fine-tune toàn bộ tham số có thể học, trừ DFL cố định của framework; số tham số thực trong phụ lục layers. Không có train classification Accuracy.

Model chọn tại epoch {selected['epoch']} theo macro F1 supported tại confidence .25, tie Accuracy rồi epoch sớm hơn. Native best.pt vẫn chọn theo native fitness. Sau khi khóa model, sweep confidence chỉ trên validation, tie F1/Accuracy rồi threshold cao hơn.

## Kết quả thực đo

| Chỉ số | Theo dõi cố định .25 | Threshold chọn {m['threshold']:.2f} |
|---|---:|---:|
| Accuracy toàn bộ positive | {fixed['accuracy']:.6f} | {m['accuracy']:.6f} |
| Macro F1 supported (28 lớp) | {fixed['val_macro_f1_supported']:.6f} | {m['val_macro_f1_supported']:.6f} |
| Weighted F1 | {fixed['weighted_f1']:.6f} | {m['weighted_f1']:.6f} |
| Coverage | {fixed['coverage']:.6f} | {m['coverage']:.6f} |

Accuracy: {m['correct']}/{m['n_positive']}; Wilson 95% [{interval[0]:.6f}, {interval[1]:.6f}]. Background: {m['background_correct_rejections']}/{m['n_background']} từ chối đúng. Conditional Accuracy: {m['accepted_accuracy']}. `nothing` positive support=0, recall/F1=N/A; FP trên positive={m['per_class']['nothing']['fp']}, FP trên background={m['per_class']['nothing']['background_fp']}.

Detection validator: {json.dumps(detection,ensure_ascii=False)}. Đây là độ phù hợp với weak localization labels tự sinh; không gọi mAP hoặc Precision là Accuracy.

## Phân tích và giới hạn

M/N: {json.dumps(m['mn_confusions'])}. Top cặp nhầm thực: {json.dumps(m['top_confusions'],ensure_ascii=False)}. Gallery ghi ID, GT box tự động, prediction, confidence và nhóm lỗi; không suy nguyên nhân chắc chắn chỉ từ một ảnh.

Cả checkpoint và threshold được chọn trên validation nên kết quả là phát triển, chưa phải đánh giá độc lập. NO_DETECTION là trạng thái prediction, không là lớp GT trong macro average. Background no_hand không nhập thành positive nothing. Frame tương quan làm yếu giả định độc lập Wilson; chống trùng không tự chứng minh signer-independent. J/Z chỉ là tư thế tĩnh. Chưa so sánh/xếp hạng với MLP hoặc ST-Hybrid trên phạm vi khác.

## Bước tiếp theo

Final test vẫn khóa. Cần ground truth phù hợp, danh sách đầy đủ và protocol chung trước tác vụ test riêng; người mới và hai điều kiện nền/ánh sáng là kiểm chứng độc lập. 3 seed/ablation/McNemar/mirror mới là đề xuất thí nghiệm tiếp theo.

Benchmark chưa thực hiện trong lệnh analysis. Lệnh benchmark riêng đo batch1, warmup20, 500 lượt trên cùng GPU/backend, tách forward YOLO khỏi tiền xử lý/NMS/MediaPipe/camera/display; không suy webcam FPS Windows.

## Phụ lục kỹ thuật

Fingerprint: {cfg['fingerprint']}. SHA model: {read_json(run/'inference.json')['checkpoint_sha256']}. Model inference: checkpoints/best_macro_f1.pt. Resume: checkpoints/last_resume.pt hoặc generation immutable trong commits. Toàn bộ history/metrics/predictions/figures/hyperparameters và checksum ở thư mục run. Checkpoint resume giữ model/EMA/optimizer/scheduler/scaler/RNG; chỉ khôi phục tới epoch hoàn tất gần nhất, không hứa bit-for-bit.
'''
    (run/'REPORT.md').write_text(text,encoding='utf-8')
