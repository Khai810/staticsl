"""Benchmark bổ sung, chỉ sau khi checkpoint/threshold được khóa."""
from pathlib import Path
import time
import numpy as np
from ..core import file_hash, read_json, read_rgb, write_json
from .reporting import write_csv
from .storage import RunLock


def benchmark_forward(root: Path, run: Path, cfg: dict) -> dict:
    """20 warmup + 500 forward batch1; không gồm MP, camera hay NMS."""
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; benchmark not executed')
    with RunLock(run):
        lock=read_json(run/'inference.json')
        checkpoint=run/lock['checkpoint']
        if file_hash(checkpoint)!=lock['checkpoint_sha256'] or lock['fingerprint']!=cfg['fingerprint']:
            raise ValueError('Inference lock mismatch')
        model=YOLO(str(checkpoint)).model.float().cuda().eval()
        samples=[s for s in read_json(run/'samples.json') if s['split']=='val'][:10]
        tensors=[]
        for sample in samples:
            image=LetterBox((320,320),auto=False)(image=read_rgb(root/cfg['release']/sample['image']))
            tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
            tensors.append(tensor)
        elapsed=[]
        with torch.inference_mode():
            for i in range(520):
                torch.cuda.synchronize()
                start=time.perf_counter()
                model(tensors[i%len(tensors)])
                torch.cuda.synchronize()
                if i>=20: elapsed.append((time.perf_counter()-start)*1000)
        result=dict(checkpoint_sha256=lock['checkpoint_sha256'],batch=1,warmup=20,repetitions=500,
                    mean_ms=float(np.mean(elapsed)),p95_ms=float(np.percentile(elapsed,95)),
                    sequential_throughput_per_second=float(1000/np.mean(elapsed)),gpu=torch.cuda.get_device_name(0),
                    backend='PyTorch eager float32 unfused; preloaded CUDA tensors',
                    scope='synchronized YOLO forward only; excludes preprocess, H2D, NMS, MediaPipe, camera/display',
                    comparison='YOLO only; not a three-model comparison and not Windows webcam FPS')
        write_json(run/'metrics/benchmark.json',result)
        write_csv(run/'metrics/benchmark_samples.csv',[dict(iteration=i+1,forward_ms=ms) for i,ms in enumerate(elapsed)])
        return result
