"""Adapter pin 8.4.178; không sao chép vòng lặp train của framework."""
from __future__ import annotations
from copy import copy, deepcopy
from pathlib import Path
import random
import time
import numpy as np
from ..core import digest, read_json, write_json
from ..training import assert_mapping
from .metrics import score_predictions
from .dataset import audit_transforms
from .storage import ControlledStop, commit_epoch, utc


def optimizer_summary(state: dict) -> dict:
    """Chữ ký tensor optimizer dùng đối chiếu trước/sau resume."""
    import torch
    tensors = [v.detach().cpu() for s in state['state'].values() for v in s.values() if isinstance(v,torch.Tensor)]
    return dict(state_entries=len(state['state']), tensors=len(tensors),
                tensor_elements=sum(t.numel() for t in tensors),
                sum_abs=sum(float(t.double().abs().sum()) for t in tensors))


def cpu_tree(value):
    """Chụp tensor CPU mà không đổi dtype optimizer."""
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(v) for v in value)
    return deepcopy(value)


def capture_rng(trainer) -> dict:
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                loader=trainer.train_loader.generator.get_state() if trainer.train_loader.generator else None)


def restore_rng(trainer, state: dict) -> None:
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    if state['loader'] is not None and trainer.train_loader.generator is not None:
        trainer.train_loader.generator.set_state(state['loader'])


def validator_class() -> type:
    """Tái dùng đúng NMS/prediction của validator, không inference trùng."""
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class ReleaseValidator(DetectionValidator):
        def run_callbacks(self, event: str) -> None:
            # This workflow is offline. Optional third-party tracking hooks are not executed.
            pass

        def init_metrics(self, model) -> None:
            self.image_predictions = {}
            super().init_metrics(model)

        def update_metrics(self, preds: list, batch: dict) -> None:
            for i,pred in enumerate(preds):
                prepared = self._prepare_batch(i,batch)
                scaled = self.scale_preds(self._prepare_pred(pred),prepared)
                sid = Path(batch['im_file'][i]).stem
                if sid in self.image_predictions:
                    raise ValueError('Repeated validation sample')
                detections = [dict(class_id=int(c), confidence=float(p), box=list(map(float,b)))
                    for c,p,b in zip(scaled['cls'].cpu().tolist(),scaled['conf'].cpu().tolist(),scaled['bboxes'].cpu().tolist())]
                # Top two suffice for highest-confidence classification at EVERY threshold
                # and the multiple-detection flag. Detection mAP still uses ALL predictions below.
                self.image_predictions[sid] = sorted(detections,key=lambda d:d['confidence'],reverse=True)[:2]
            super().update_metrics(preds,batch)
    return ReleaseValidator


def trainer_class(context: dict) -> type:
    """Tạo trainer lưu raw model/EMA/optimizer/scheduler/RNG từng epoch."""
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics import __version__
    if __version__ != '8.4.178':
        raise RuntimeError('This adapter has only been inspected for Ultralytics 8.4.178')
    run, cfg = Path(context['run']), context['cfg']

    class ReleaseTrainer(DetectionTrainer):
        def run_callbacks(self, event: str) -> None:
            # Only our local hooks run; final_eval's extra fit callback is intentionally ignored.
            if event == 'on_train_start':
                self._release_train_start()
            elif event == 'on_train_epoch_start':
                self.release_epoch_start = time.perf_counter()
                self.release_gradient_steps = 0
                torch.cuda.reset_peak_memory_stats(self.device)
            elif event == 'on_train_batch_end':
                if not torch.isfinite(self.loss).all():
                    raise FloatingPointError('Nonfinite training loss; no synthetic recovery metrics')

        def _build_train_pipeline(self) -> None:
            if self.batch_size != cfg['batch']:
                # Native first-epoch OOM retry would retain partial optimizer state.
                raise torch.cuda.OutOfMemoryError('Native batch auto-reduction disabled; create NEW pretrained attempt')
            super()._build_train_pipeline()

        def _handle_nan_recovery(self, epoch: int) -> bool:
            if not torch.isfinite(self.loss).all() or any(not np.isfinite(float(x)) for x in self.metrics.values()):
                raise FloatingPointError('Nonfinite loss/validation; preserve failed attempt')
            return False

        def setup_model(self):
            ckpt = super().setup_model()
            if self.resume:
                if ckpt is None or ckpt.get('raw_model') is None:
                    raise ValueError('Need full release resume checkpoint, not inference weights')
                self.model.load_state_dict(ckpt['raw_model'],strict=True)
            return ckpt

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg,weights,verbose)
            if weights is not None and not self.resume:
                target, source = model.state_dict(), weights.float().state_dict()
                matched = [k for k,v in target.items() if k in source and v.shape == source[k].shape]
                write_json(run/'pretrained_transfer.json',dict(matched_tensors=len(matched),total_target_tensors=len(target),
                           matched_elements=sum(target[k].numel() for k in matched),
                           matched_keys=matched,unmatched_keys=[k for k in target if k not in matched],
                           source='verified COCO assets/yolo11n.pt; detector rebuilt with 29 outputs'))
            return model

        def get_validator(self):
            return validator_class()(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks={})

        def _release_train_start(self) -> None:
            assert_mapping(self.model.names,context['classes'])
            for key,value in context['augmentation'].items():
                if getattr(self.args,key) != value:
                    raise ValueError(f'Resolved augmentation changed: {key}')
            if self.test_loader.dataset.augment or not self.train_loader.dataset.augment:
                raise ValueError('Unexpected loader augmentation flag')
            write_json(run/'loader_augmentation.json',dict(
                train=audit_transforms(self.train_loader.dataset.transforms,context['augmentation'],True),
                val=audit_transforms(self.test_loader.dataset.transforms,context['augmentation'],False)))
            write_json(run/'resolved_config.json',vars(self.args))
            params = [dict(name=n,shape=list(p.shape),elements=p.numel(),trainable=p.requires_grad)
                      for n,p in self.model.named_parameters()]
            write_json(run/'layers.json',dict(parameters=params,trainable_elements=sum(p['elements'] for p in params if p['trainable']),
                       frozen_elements=sum(p['elements'] for p in params if not p['trainable']),
                       note='freeze=0 fine-tunes all learnable weights; native fixed DFL remains frozen'))
            write_json(run/'optimizer.json',dict(type=type(self.optimizer).__name__,
                       param_groups=[{k:v for k,v in g.items() if k!='params'} for g in self.optimizer.param_groups],
                       scheduler=type(self.scheduler).__name__,state=self.scheduler.state_dict()))
            resumed = context.get('resume_ckpt')
            if resumed is not None:
                self.scheduler.load_state_dict(resumed['scheduler'])
                actual = optimizer_summary(self.optimizer.state_dict())
                expected = optimizer_summary(resumed['optimizer'])
                if self.start_epoch != resumed['epoch']+1 or actual != expected:
                    raise ValueError('Optimizer or start epoch was not restored')
                restore_rng(self,resumed['rng'])
                write_json(run/f'resume_restored_epoch_{self.start_epoch:04d}.json',dict(
                    at_utc=utc(),start_epoch_zero_based=self.start_epoch,next_epoch_one_based=self.start_epoch+1,
                    optimizer_actual=actual,optimizer_expected=expected,optimizer_verified=True,
                    scheduler_restored=True,raw_model_restored=True,ema_updates=self.ema.updates,
                    rng_restored=True,limit='not bit-for-bit: loader iterator, augmentation, device kernels may differ'))
            self.release_started = time.perf_counter()

        def optimizer_step(self) -> None:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError('Missing/nonfinite gradients')
            self.release_gradient_steps += 1
            super().optimizer_step()

        def save_model(self) -> bool:
            if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
                raise FloatingPointError('Nonfinite model; refuse checkpoint')
            if self.release_gradient_steps < 1:
                raise ValueError('Epoch has no verified optimizer step')
            super().save_model()  # native last/best/epoch retain original native fitness meaning
            predictions = self.validator.image_predictions
            image_metrics = score_predictions(context['val_samples'],predictions,context['classes'],
                                             context['supported'],cfg['monitor_confidence'])
            epoch = self.epoch+1
            history = read_json(run/'metrics/history.json') if (run/'metrics/history.json').exists() else []
            elapsed = time.perf_counter()-self.release_epoch_start
            metrics = {**self.label_loss_items(self.tloss),**self.metrics,**self.lr,
                       **image_metrics, 'epoch':epoch,'epoch_seconds':elapsed,
                       'total_seconds':sum(h['epoch_seconds'] for h in history)+elapsed,
                       'peak_gpu_bytes':int(torch.cuda.max_memory_allocated(self.device)),
                       'verified_optimizer_steps':self.release_gradient_steps, 'native_fitness':float(self.fitness),
                       'checkpoint_role':'EMA float32 validation; raw model separately retained for resume'}
            if any(not np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float))):
                raise FloatingPointError('Nonfinite epoch metrics')
            ckpt = torch.load(self.last,map_location='cpu',weights_only=False)
            ema_snapshot = deepcopy(self.ema.ema).float().cpu()
            ema_snapshot.criterion = None
            ckpt.update(raw_model=cpu_tree(unwrap_model(self.model).state_dict()),
                        ema=ema_snapshot,
                        optimizer=cpu_tree(self.optimizer.state_dict()),scheduler=deepcopy(self.scheduler.state_dict()),
                        scaler=cpu_tree(self.scaler.state_dict()),rng=capture_rng(self),
                        release_context=dict(fingerprint=cfg['fingerprint'],config_sha256=digest(cfg),
                                             classes=context['classes'],source_hashes=context['source_hashes']))
            record = commit_epoch(run,ckpt,metrics,predictions)
            print(f"EPOCH {epoch} COMMITTED | val Accuracy={image_metrics['accuracy']:.4f} | "
                  f"macro F1 supported({len(context['supported'])})={image_metrics['val_macro_f1_supported']:.4f} | "
                  f"mAP50={self.metrics.get('metrics/mAP50(B)')} | {elapsed:.1f}s",flush=True)
            if context.get('stop_after') == epoch:
                raise ControlledStop(f'Epoch {epoch} committed; controlled stop before final_eval')
            return True
    return ReleaseTrainer
