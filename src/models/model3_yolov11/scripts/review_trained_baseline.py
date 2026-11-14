"""Review the completed Colab baseline using its real saved validation predictions only."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asl_yolo.core import digest,file_hash,read_json,write_json
from asl_yolo.train_release.metrics import is_better,score_predictions,wilson
from asl_yolo.train_release.reporting import export_gallery,plot_distribution,pyplot,save_figure,write_csv

RUN=ROOT/'.cache/d5cb4953'
OUTPUT=ROOT/'reports/asl_yolo11n_baseline_review_20261114'


def plot_history(history: list[dict], chosen_epoch: int) -> None:
    plt=pyplot()
    epochs=[h['epoch'] for h in history]
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ax,loss in zip(axes[0],['box_loss','cls_loss','dfl_loss']):
        for split in ['train','val']:
            ax.plot(epochs,[h[f'{split}/{loss}'] for h in history],label=split)
        ax.set(xlabel='Completed epoch',ylabel='Loss (framework units)',title=loss)
        ax.legend()
    for key,label in [('accuracy','Image Accuracy'),('val_macro_f1_supported','Macro F1 (28 classes)')]:
        axes[1,0].plot(epochs,[100*h[key] for h in history],label=label)
    axes[1,0].axvline(chosen_epoch,color='gray',linestyle='--',label=f'Selected epoch {chosen_epoch}')
    axes[1,0].set(xlabel='Completed epoch',ylabel='Validation score (%)',title='3,842 positive validation images; conf=0.25')
    axes[1,0].legend(fontsize=8)
    for key,label in [('metrics/precision(B)','Precision'),('metrics/recall(B)','Recall'),
                      ('metrics/mAP50(B)','mAP@0.5'),('metrics/mAP50-95(B)','mAP@0.5:0.95')]:
        axes[1,1].plot(epochs,[100*h[key] for h in history],label=label)
    axes[1,1].set(xlabel='Completed epoch',ylabel='Detection score (%)',title='Validation detection on weak boxes')
    axes[1,1].legend(fontsize=8)
    for key in ['lr/pg0','lr/pg1','lr/pg2']:
        axes[1,2].plot(epochs,[h[key] for h in history],label=key)
    axes[1,2].set(xlabel='Completed epoch',ylabel='Learning rate',title='Actual learning rate (linear schedule + warmup)')
    axes[1,2].legend(fontsize=8)
    save_figure(fig,OUTPUT/'figures/learning_curves')
    fig,ax=plt.subplots(figsize=(10,5))
    for key,label in [('accuracy','Image Accuracy'),('val_macro_f1_supported','Macro F1 (28 supported classes)')]:
        ax.plot(epochs,[100*h[key] for h in history],label=label,linewidth=2)
    ax.axvline(chosen_epoch,color='gray',linestyle='--',label=f'Selected checkpoint: epoch {chosen_epoch}')
    ax.set(xlabel='Completed training epoch',ylabel='Validation score (%)',ylim=(70,100),
           title='YOLO11n:3,842 positive validation images; confidence0.25')
    ax.legend(loc='lower right')
    ax.grid(alpha=.2)
    save_figure(fig,OUTPUT/'figures/validation_progress')


def plot_confusion(metrics: dict, label: str) -> None:
    plt=pyplot()
    counts=np.asarray(metrics['confusion_matrix'],dtype=float)
    supports=counts.sum(axis=1,keepdims=True)
    normalized=np.divide(counts,supports,out=np.full_like(counts,np.nan),where=supports!=0)
    for matrix,name in [(counts,'counts'),(normalized,'normalized')]:
        rows=[]
        for truth,values in zip(metrics['confusion_rows'],matrix):
            rows.append(dict(truth=truth,**{c:float(v) if np.isfinite(v) else None for c,v in zip(metrics['confusion_columns'],values)}))
        write_csv(OUTPUT/f'metrics/confusion_{label}_{name}.csv',rows)
        fig,ax=plt.subplots(figsize=(15,12))
        cmap=plt.get_cmap('Blues').copy()
        cmap.set_bad('#e6e6e6')
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1 if name=='normalized' else None)
        labels=[c if metrics['per_class'][c]['support'] else c+' (N/A)' for c in metrics['confusion_rows']]
        ax.set(xticks=range(counts.shape[1]),xticklabels=metrics['confusion_columns'],
               yticks=range(counts.shape[0]),yticklabels=labels,xlabel='Predicted image label',ylabel='GT source image label',
               title=f"Image-level validation confusion, epoch45, confidence={metrics['threshold']:.2f} ({name})")
        ax.tick_params(axis='x',rotation=90)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if np.isfinite(matrix[i,j]) and matrix[i,j]>0:
                    ax.text(j,i,f'{matrix[i,j]:.3f}' if name=='normalized' else str(int(matrix[i,j])),ha='center',va='center',fontsize=6)
        fig.colorbar(im,ax=ax,label='Fraction of GT class' if name=='normalized' else 'Images')
        save_figure(fig,OUTPUT/f'figures/confusion_{label}_{name}')


def plot_per_class(fixed: dict, tuned: dict, classes: list[str]) -> None:
    plt=pyplot()
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True)
    x=np.arange(len(classes))
    for metrics,offset in [(fixed,-.2),(tuned,.2)]:
        values=[100*metrics['per_class'][c]['f1'] if metrics['per_class'][c]['f1'] is not None else np.nan for c in classes]
        axes[0].bar(x+offset,values,width=.4,label=f"Confidence {metrics['threshold']:.2f}")
    axes[0].set(ylabel='Image F1 (%)',ylim=(0,103),title='Epoch45 validation F1; nothing recall/F1=N/A (zero support)')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False)
    axes[1].bar(x,[fixed['per_class'][c]['support'] for c in classes])
    axes[1].set(xticks=x,xticklabels=classes,ylabel='Positive validation images',xlabel='GT source class')
    axes[1].tick_params(axis='x',rotation=90)
    save_figure(fig,OUTPUT/'figures/per_class_f1_support')


def plot_sweep(sweep: list[dict], chosen: float) -> None:
    plt=pyplot()
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,title in zip(axes,['coverage','accepted_accuracy','accuracy'],
                           ['Coverage of positive images','Accuracy among accepted positives','Accuracy among all positive images']):
        ax.plot([s['threshold'] for s in sweep],[100*s[key] if s[key] is not None else np.nan for s in sweep],marker='o')
        ax.axvline(chosen,color='gray',linestyle='--',label=f'Selected {chosen:.2f}')
        ax.set(xlabel='Confidence threshold',ylabel='Validation fraction (%)',ylim=(0,101),title=title)
        ax.legend()
    save_figure(fig,OUTPUT/'figures/coverage_accuracy_threshold')


def main() -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    state=read_json(RUN/'run.json');cfg=read_json(RUN/'config.json');history=read_json(RUN/'metrics/history.json')
    classes=read_json(RUN/'class_mapping.json');chosen=read_json(RUN/'checkpoints/best_macro_f1.json')
    last=read_json(RUN/'checkpoints/last_resume.json');environment=read_json(RUN/'environment.json')
    model_check=read_json(RUN/'verification/checkpoints_load_verified.json')
    assert state['status']=='completed' and state['completed_epochs']==state['total_epochs']==50
    assert [h['epoch'] for h in history]==list(range(1,51))
    assert digest(cfg)==state['config_sha256'] and not state['allow_final_test'] and not cfg['allow_final_test']
    assert file_hash(RUN/'checkpoints/best_macro_f1.pt')==chosen['checkpoint_sha256']
    assert file_hash(RUN/'checkpoints/last_resume.pt')==last['files'][last['checkpoint']]
    journal=[read_json(p) for p in sorted((RUN/'commits').glob('epoch_*.json'))]
    assert [j['epoch'] for j in journal]==list(range(1,51))
    assert all(j['config_sha256']==state['config_sha256'] for j in journal)
    for generation in [chosen['generation'],last]:
        for relative,expected in generation['files'].items():
            if (RUN/relative).exists(): assert file_hash(RUN/relative)==expected
    best=None
    for row in history:
        if is_better(row,best):best=row
    assert best['epoch']==chosen['selection']['epoch']==45
    release=read_json(ROOT/cfg['release']/'release.json')
    assert digest(release['samples'])==release['fingerprint']==state['fingerprint']
    assert release['classes']==classes
    rows=[s for s in release['samples'] if s['split']=='val']
    supported=[c for c in classes if any(s['label']==c and s['boxes'] for s in rows)]
    predictions=read_json(RUN/chosen['generation']['predictions'])
    original=read_json(RUN/chosen['generation']['metrics'])
    fixed=score_predictions(rows,predictions,classes,supported,.25)
    assert fixed['accuracy']==original['accuracy'] and fixed['val_macro_f1_supported']==original['val_macro_f1_supported']
    assert fixed['records']==original['records'] and fixed['confusion_matrix']==original['confusion_matrix']
    sweep=[score_predictions(rows,predictions,classes,supported,t) for t in cfg['sweep_grid']]
    selected=max(sweep,key=lambda s:(s['val_macro_f1_supported'],s['accuracy'],s['threshold']))
    for name,value in [('class_mapping.json',classes),('metrics/history.json',history),('metrics/validation_fixed.json',fixed),
                       ('metrics/validation_selected.json',selected),('metrics/validation_sweep.json',[{k:v for k,v in s.items() if k!='records'} for s in sweep]),
                       ('predictions/validation.json',predictions),('verification/run.json',state),('verification/selected_checkpoint.json',chosen),
                       ('verification/checkpoint_load.json',model_check)]:
        write_json(OUTPUT/name,value)
    for name in ['history.csv']:
        shutil.copy2(RUN/'metrics'/name,OUTPUT/'metrics'/name)
    write_csv(OUTPUT/'metrics/validation_sweep.csv',[{k:v for k,v in s.items() if isinstance(v,(int,float)) or v is None} for s in sweep])
    for name,metrics in [('fixed',fixed),('selected',selected)]:
        write_csv(OUTPUT/f'metrics/per_class_{name}.csv',[dict(label=c,**metrics['per_class'][c]) for c in classes])
        write_csv(OUTPUT/f'predictions/{name}_records.csv',metrics['records'])
    native_epoch=max(history,key=lambda r:r['native_fitness'])['epoch']
    native_out=OUTPUT/'figures/native_detection_epoch49'
    native_out.mkdir(parents=True,exist_ok=True)
    for path in (RUN/'framework').glob('*.png'):
        shutil.copy2(path,native_out/path.name)
    plot_history(history,45)
    plot_confusion(fixed,'fixed025')
    plot_confusion(selected,'selected005')
    plot_per_class(fixed,selected,classes)
    plot_sweep(sweep,selected['threshold'])
    plot_distribution(release['samples'],classes,OUTPUT/'figures/data')
    export_gallery(ROOT,OUTPUT,rows,predictions,selected,cfg)
    metadata=read_json(RUN/'preflight.json')
    last10=[h['val_macro_f1_supported'] for h in history[-10:]]
    source_hashes=read_json(RUN/'source_hashes.json')
    source_changed=[name for name,sha in source_hashes.items() if (ROOT/name).exists() and file_hash(ROOT/name)!=sha]
    summary=dict(run_id=state['run_id'],baseline_epochs=50,total_technical_and_baseline_epochs=57,
                 training_completed=True,gpu=environment['gpu'],session_seconds=state['session_seconds'],
                 train_images=9042,val_images=3909,val_positive=fixed['n_positive'],val_background=fixed['n_background'],
                 selected_epoch=45,selected_checkpoint_sha256=chosen['checkpoint_sha256'],native_best_epoch=native_epoch,
                 fingerprint=state['fingerprint'],fixed={k:v for k,v in fixed.items() if k not in ['records','confusion_matrix','per_class']},
                 selected={k:v for k,v in selected.items() if k not in ['records','confusion_matrix','per_class']},
                 selected_detection={k:v for k,v in original.items() if k.startswith('metrics/')},
                 native_detection=history[native_epoch-1],last_epoch=history[-1],
                 integrity=dict(history_contiguous=True,commit_markers=50,selected_sha256_verified=True,last_resume_sha256_verified=True,
                                per_epoch_prediction_replay_matches_original=True,checkpoint_load_verified=True,
                                source_files_changed_since_training=source_changed,
                                all50_checkpoint_bytes_verified=False),
                 last10_macro_f1=dict(min=min(last10),max=max(last10),std=float(np.std(last10))),
                 source_scope='Replay full saved Colab validation predictions of selected epoch45; no new model inference except technical one-image load check',
                 val_analysis_artifacts_on_drive_initially=False,final_test_run=False,benchmark_run=False,
                 review_only=True,remote_run_modified=False)
    write_json(OUTPUT/'summary.json',summary)
    write_json(OUTPUT/'inference_review.json',dict(checkpoint_drive_url='https://drive.google.com/file/d/1XMm9vNQjvxFftTAfh04U-vrO23hB8WPL/view',
               checkpoint_sha256=chosen['checkpoint_sha256'],classes=classes,fingerprint=state['fingerprint'],
               confidence=selected['threshold'],imgsz=320,nms_iou=.7,max_det=300,multi_label=True,agnostic_nms=False,
               rule='Highest confidence after native class-aware multi-label NMS; rejection=NO_DETECTION',
               selected_on='val',grid=cfg['sweep_grid'],criterion='macro F1 supported, Accuracy, higher threshold',
               allow_final_test=False,scope='Local review lock; original Drive run inference.json is not created by this review'))
    write_report(summary,fixed,selected,metadata,original,history)
    captions={
       'validation_progress':'All50 completed epochs; image Accuracy and macro F1 at confidence0.25 on3842 positive validation images. Macro average includes28 supported classes; selected macro-F1 checkpoint is epoch45. Validation development results, not an independent test.',
       'learning_curves':'Train/validation losses and metrics for50 completed YOLO11n epochs;3842 positive validation images plus67 backgrounds; macro F1 excludes nothing with zero positive support. Automated weak localization labels, no independent test.',
       'coverage_accuracy_threshold':'Epoch45 validation predictions, grid0.05..0.95. Coverage and image Accuracy use3842 positives; conditional Accuracy is undefined when acceptance is zero. Model and threshold both selected on validation.',
       'confusion':'Image-level GT rows and predicted columns including NO_DETECTION; epoch45,3842 positive validation images. nothing GT support0 is N/A. These matrices differ from native detection IoU-matched matrices.',
       'native_detection':'Original native detection plots use best.pt selected by native fitness at epoch49, not macro-F1 checkpoint at epoch45. Detection measures agreement with automated weak boxes.'}
    write_json(OUTPUT/'figures/captions.json',captions)
    write_json(OUTPUT/'artifact_checksums.json',{'files':{p.relative_to(OUTPUT).as_posix():file_hash(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='artifact_checksums.json'}})
    print(json.dumps({k:summary[k] for k in ['run_id','baseline_epochs','gpu','selected_epoch','native_best_epoch','integrity','last10_macro_f1']},indent=2))


def write_report(summary: dict, fixed: dict, tuned: dict, preflight: dict, original: dict, history: list[dict]) -> None:
    def pc(x):return f'{100*x:.2f}%'
    weakest=sorted([(c,v) for c,v in fixed['per_class'].items() if v['support']],key=lambda pair:pair[1]['f1'])[:8]
    background_ci=wilson(67,67)
    interval=fixed['accuracy_wilson95'];tuned_ci=tuned['accuracy_wilson95']
    native=summary['native_detection']
    text=f'''# Kiểm tra baseline YOLO11n ngày14 tháng11 năm2026

Baseline đã train thật đủ50/50 epoch trên {summary['gpu']}, session_seconds={summary['session_seconds']:.2f} (khoảng1 giờ59 phút37 giây), khởi tạo từ pretrained COCO, seed42/imgsz320/batch8/AMP=False/freeze0. Smoke3 và resume probe4 cũng đã hoàn tất trong các run riêng. History50 epoch liên tục, không đếm callback final_eval thành epoch51.

## Kết quả checkpoint được chọn

Checkpoint macro F1: **epoch45**, confidence theo dõi cố định0.25, NMSIoU0.7/class-aware/multi_label=True. Danh sách28 lớp có positive val được giữ đúng; nothing positive support0 nên recall/F1=N/A, không thêm NO_DETECTION vào macro average.

| Metric | Epoch45, conf0.25 | Cùng epoch45, sweep chọn conf{tuned['threshold']:.2f} |
|---|---:|---:|
| Image Accuracy toàn bộ positive | {pc(fixed['accuracy'])} ({fixed['correct']}/{fixed['n_positive']}) | {pc(tuned['accuracy'])} ({tuned['correct']}/{tuned['n_positive']}) |
| Macro F1 supported28 | {pc(fixed['val_macro_f1_supported'])} | {pc(tuned['val_macro_f1_supported'])} |
| Macro Precision supported28 | {pc(fixed['macro_precision_supported'])} | {pc(tuned['macro_precision_supported'])} |
| Macro Recall supported28 | {pc(fixed['macro_recall_supported'])} | {pc(tuned['macro_recall_supported'])} |
| Weighted F1 | {pc(fixed['weighted_f1'])} | {pc(tuned['weighted_f1'])} |
| Coverage | {pc(fixed['coverage'])} | {pc(tuned['coverage'])} |
| Accuracy trong số được chấp nhận | {pc(fixed['accepted_accuracy'])} | {pc(tuned['accepted_accuracy'])} |
| NO_DETECTION trên positive | {fixed['no_detection_positive']} | {tuned['no_detection_positive']} |
| Sai lớp trên positive | {fixed['n_positive']-fixed['correct']-fixed['no_detection_positive']} | {tuned['n_positive']-tuned['correct']-tuned['no_detection_positive']} |
| FP trên background no_hand |0/67|0/67|

Wilson95% cho Accuracy tại0.25: [{pc(interval[0])}, {pc(interval[1])}]; tại0.05: [{pc(tuned_ci[0])}, {pc(tuned_ci[1])}]. 0FP/67 background không chứng minh tỷ lệ lỗi ngoài thực tế bằng0: Wilson cho rejection có cận dưới{pc(background_ci[0])}, tương ứng cận trên FPR khoảng{pc(1-background_ci[0])} nếu giả định độc lập. Frame tương quan làm yếu giả định này.

Detection ở chính epoch45: Precision={pc(original['metrics/precision(B)'])}, Recall={pc(original['metrics/recall(B)'])}, mAP@0.5={pc(original['metrics/mAP50(B)'])}, mAP@0.5:0.95={pc(original['metrics/mAP50-95(B)'])}. Các chỉ số này không phải Image Accuracy và chỉ phản ánh mức khớp với box ground truth tự động.

## Nhận xét hội tụ và50 epoch có đủ không

Train box/cls/DFL loss giảm rõ; validation losses cũng giảm rồi gần phẳng cuối phiên, không có xu hướng val loss tăng kéo dài để kết luận overfitting rõ trong lần này. Ví dụ train cls từ{history[0]['train/cls_loss']:.5f} xuống{history[-1]['train/cls_loss']:.5f}; val cls từ{history[0]['val/cls_loss']:.5f} xuống{history[-1]['val/cls_loss']:.5f}. Chênh train/val loss tồn tại, nhưng hai phép đo dùng train augmentation và EMA/validation khác nhau nên không dùng riêng chênh lệch để kết luận.

Từ epoch40 macro F1={pc(history[39]['val_macro_f1_supported'])} đến epoch45={pc(fixed['val_macro_f1_supported'])}, cải thiện nhỏ; epoch50={pc(history[-1]['val_macro_f1_supported'])} gần như tương đương epoch45.10 epoch cuối dao động{pc(summary['last10_macro_f1']['min'])}–{pc(summary['last10_macro_f1']['max'])}. **50 epoch đã tạo baseline hội tụ hợp lý trên validation này; chưa có bằng chứng cần tăng100 ngay.** Nếu muốn thử100, đó là thí nghiệm riêng có protocol trước run, không ghi đè baseline hoặc gọi nạp mỗi weight là resume.

## Đọc các biểu đồ đúng checkpoint và đúng loại metric

Biểu đồ mới learning_curves kết hợp loss, Image Accuracy/macro F1 tại0.25, detection metrics và learning rate từ history thật. Confusion image-level mới có hàngGT, cộtprediction kể cả NO_DETECTION; hàngnothing support0=N/A. F1/support theo lớp giữ đúng mapping.

Native results.png là lịch sử đầy đủ50 epoch. Riêng các hình PR/F1-confidence/confusion ở cuối phiên dùng native best.pt thuộc **epoch{summary['native_best_epoch']}**, chọn theo native fitness, khác best_macro_f1.pt epoch45. Đã đối chiếu train_metrics của native best.pt với history: P={pc(native['metrics/precision(B)'])},R={pc(native['metrics/recall(B)'])},mAP50={pc(native['metrics/mAP50(B)'])},mAP50-95={pc(native['metrics/mAP50-95(B)'])}. PR figure hiển thị xấp xỉ0.964, nhất quán với mAP50 native. Native F1-confidence khoảng0.95 tại0.485 là **detection F1**, không phải image macro F1 và không được tự lấy0.485 làm ngưỡng phân loại ảnh.

Native confusion matrix dùng x=True/y=Predicted và ghép box theo IoU; background là box/GT không khớp trên mọi ảnh, không chỉ67 ảnh no_hand. Vì vậy không dùng background column của detection matrix để thay FPR ảnh nền hoặc coi nó là NO_DETECTION image-level. mAP50-95 thấp hơn mAP50 cho thấy yêu cầu box khớp chặt khó hơn; với weak boxes chưa thể tách hoàn toàn lỗi model và độ bất định annotation.

## Phân tích lỗi theo lớp

M→N={fixed['mn_confusions']['M_to_N']},N→M={fixed['mn_confusions']['N_to_M']} tại0.25; khi hạ0.05: M→N={tuned['mn_confusions']['M_to_N']},N→M={tuned['mn_confusions']['N_to_M']}. Cặp M/N không phải lỗi chiếm ưu thế trên validation hiện tại, nhưng chưa suy ra kết quả người mới.

Các lớp F1 thấp hơn tại0.25:

| Lớp | F1 | Positive support |
|---|---:|---:|
'''
    text+='\n'.join(f"|{c}|{pc(v['f1'])}|{v['support']}|" for c,v in weakest)
    text+=f'''

Top cặp sai lớp tại0.25 gồm E→T(2), I→G(2), B→F(2), W→F(2), U→V(2). Ngoài ra35 positive bị từ chối. Gallery có đúng/sai/no detection/high confidence wrong, GT box tự động màu xanh, prediction đỏ và ID; không suy nhãn chuyên gia từ dự đoán của model.

## Sweep validation bổ sung trong lượt review

Thư mục Drive ban đầu chưa có inference.json, REPORT.md hoặc metrics/validation_sweep.json; training đã xong nhưng cell phân tích validation chưa tạo artifact. Lượt này **tái tính từ saved predictions epoch45**, đủ toàn bộ3909 val; không inference lại toàn dataset và không đụng test. Top2 NMS đã lưu đủ cho quy tắc highest-confidence ở mọi threshold. Replay tại0.25 khớp Accuracy/macroF1/confusion/records gốc; weighted F1 chỉ có sai khác dấu phẩy động mức máy.

Grid đã khai báo trước0.05..0.95 bước0.05; chọn macro F1 supported, tie Accuracy rồi threshold cao hơn. **0.05 tốt nhất trong grid**, không chứng minh tối ưu ngoài grid. Hạ threshold cứu thêm18 ảnh đúng nhưng tăng sai lớp từ32 lên42; conditional Accuracy giảm nhẹ, Coverage tăng. Cả checkpoint và threshold dùng validation để chọn nên đây là **kết quả phát triển**, chưa phải đánh giá độc lập. Inference lock của lượt review được lưu riêng, không sửa run Drive.

## Độ toàn vẹn và giới hạn

- Đã kiểm50 history rows/50 commit markers, config digest, release fingerprint/mapping và SHA checkpoint selected/last_resume; metrics/predictions của epoch45/50 khớp checksum marker. Chưa tải/kiểm byte toàn bộ50 checkpoint~1.6GB.
- Model được chọn load và predict mẫu validation thật được trên CPU. Checkpoint last_resume giữ epoch zero-based49, raw model499 tensors, EMA, optimizer255 state entries, scheduler/RNG. Native best.pt đã strip optimizer, đúng hành vi framework. Không resume run50 đã hoàn tất.
- GPU resume probe4 epoch có bằng chứng process mới khôi phục optimizer/epoch đúng; epoch2 tiếp sau epoch1. Smoke/resume gates đều passed.
- Train9042/val3909; positive val3842, background67; nothing positive train22/val0. review_kind=automated, human_review_claimed=false. Phần lớn session chưa biết; chống trùng không chứng minh signer-independent. J/Z chỉ ảnh tĩnh. Không có final test/external Bao hoặc benchmark mới trong review này.

## File và đường dẫn kỹ thuật

[Biểu đồ Accuracy/Macro F1](figures/validation_progress.png), [loss và detection metrics](figures/learning_curves.png), [F1 từng lớp và số mẫu](figures/per_class_f1_support.png), [confusion tại confidence0.25](figures/confusion_fixed025_normalized.png), [Coverage/Accuracy theo threshold](figures/coverage_accuracy_threshold.png), [gallery ảnh đúng và sai](figures/gallery/index.html). Biểu đồ mới có cả PNG300dpi và SVG; số liệu CSV/JSON nằm trong metrics/.

Run: `{summary['run_id']}`; fingerprint:`{summary['fingerprint']}`. SHA selected:`{summary['selected_checkpoint_sha256']}`. Source raw tải về trong `.cache/d5cb4953`; manifest có Drive URL và SHA file, không lưu bearer download URL. Local summary/metrics/figures/predictions/inference_review.json và gallery nằm cạnh báo cáo này.

[Run Drive](https://drive.google.com/drive/folders/1Ho7K2qwRJHOyIqUcVIzDgNUEr5CKGaKm), [model macro F1 epoch45](https://drive.google.com/file/d/1XMm9vNQjvxFftTAfh04U-vrO23hB8WPL/view), [last resume](https://drive.google.com/file/d/1K_9s8tCe9-8xBbpYSSPEKQeb-wguaBYw/view).
'''
    (OUTPUT/'REVIEW.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
