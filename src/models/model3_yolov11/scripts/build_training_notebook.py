"""Sinh notebook 03 không output, độc lập các notebook prototype."""
from pathlib import Path
import argparse
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-bundle-folder',default='REPLACE_WITH_NEW_BUNDLE_FOLDER')
    args=parser.parse_args()
    if Path(args.drive_bundle_folder).name!=args.drive_bundle_folder or args.drive_bundle_folder in ('.','..'):
        raise ValueError('Drive bundle folder must be a single folder name')
    cells=[]
    def md(text):
        cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip()+'\n'))
    def code(text):
        cells.append(dict(cell_type='code',metadata={},source=textwrap.dedent(text).strip()+'\n',outputs=[],execution_count=None))
    md('''
    # ASL YOLO11n auto V2 training

    Luồng mới 29 đầu ra, 9.042 train / 3.909 validation; 28 lớp có positive validation.
    Weak labels tự động; không annotation chuyên gia ASL. Không có final test/external/webcam trong notebook.
    Chọn **Runtime → Change runtime type → GPU**. Chạy từng phần theo thứ tự; các cell train có cờ riêng mặc định False.
    Không dùng notebook prototype 00/01 để khởi chạy release này. Khi ngắt: epoch hoàn tất gần nhất có thể cứu;
    trước epoch đầu chưa có checkpoint. Browser disconnect không tự chứng minh GPU job đã chết.
    ''')
    md('''
    ## 1 Mount Drive và kiểm tra gói thật

    Bộ chạy cần **source.zip**, **bundle.json** và **dataset.zip** hoặc toàn bộ các phần **dataset.zip.partXXX**.
    Nếu dùng các phần nhỏ, notebook tự ghép ở runtime và kiểm tra SHA-256 của ZIP gốc trước khi giải nén.
    Điền `BUNDLE_DIR` đúng thư mục đó; notebook kiểm tra tồn tại và SHA, không giả định gói mới đã ở Drive.
    `DRIVE_PROJECT` chỉ được dùng nếu MyDrive đã mount thực; source/data cache ở /content, mọi checkpoint/history ở Drive.
    ''')
    code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib, json, zipfile, os, sys, subprocess
    DRIVE_PROJECT = Path('/content/drive/MyDrive/ASL_YOLO11n')
    BUNDLE_DIR = DRIVE_PROJECT / 'exports' / 'REPLACE_WITH_NEW_BUNDLE_FOLDER'
    WORK = Path('/content/asl_auto_v2_training')
    assert Path('/content/drive/MyDrive').is_dir(), 'Drive is not mounted'
    assert (BUNDLE_DIR / 'bundle.json').is_file(), 'Upload the new package and set BUNDLE_DIR'
    manifest = json.loads((BUNDLE_DIR / 'bundle.json').read_text())
    def sha256(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    def local_archive(name):
        info = manifest[name]
        if not info.get('parts'):
            return BUNDLE_DIR / info['file']
        cache = WORK / '.bundle_cache'
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / info['file']
        if target.exists() and sha256(target) == info['sha256']:
            return target
        partial = cache / (info['file'] + '.partial')
        import shutil
        with partial.open('wb') as dst:
            for i, part in enumerate(info['parts'], 1):
                source = (BUNDLE_DIR / part['file']).resolve()
                assert source.is_relative_to(BUNDLE_DIR.resolve()), 'Unsafe part path'
                assert source.is_file(), f'Missing uploaded part: {part["file"]}'
                assert source.stat().st_size == part['bytes'], 'Part size mismatch'
                assert sha256(source) == part['sha256'], f'Part checksum differs: {part["file"]}'
                print(f'Assembling {name}: part {i}/{len(info["parts"])}', flush=True)
                with source.open('rb') as src:
                    shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        assert partial.stat().st_size == info['bytes'], 'Assembled archive size differs'
        assert sha256(partial) == info['sha256'], 'Assembled archive checksum differs'
        os.replace(partial, target)
        return target
    def restore_bundle(name, manifest_name):
        archive = local_archive(name)
        assert sha256(archive) == manifest[name]['sha256'], f'{name} archive checksum differs'
        with zipfile.ZipFile(archive) as z:
            entries = json.loads(z.read(manifest_name))['files']
            for relative, expected in entries.items():
                target = (WORK / relative).resolve()
                assert target.is_relative_to(WORK.resolve()), 'Unsafe archive path'
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert sha256(target) == expected, f'Existing cache changed: {relative}; choose a new WORK'
                else:
                    partial = target.with_name(target.name + '.partial')
                    with z.open(relative) as src, partial.open('wb') as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    assert sha256(partial) == expected, f'Extracted checksum differs: {relative}'
                    os.replace(partial, target)
    restore_bundle('source', 'source_bundle_manifest.json')
    restore_bundle('dataset', 'dataset_bundle_manifest.json')
    os.chdir(WORK)
    print('Verified new auto V2 package:', manifest['counts'], manifest['fingerprint'])
    ''')
    md('''
    ## 2 Dependencies giữ nguyên PyTorch CUDA của runtime

    Ultralytics pin 8.4.178; không cài lock CPU Windows. Các pin direct dependencies được kiểm tra local,
    chưa gọi là bộ đã train thành công trên Colab. Constraints giữ torch/torchvision; runtime thực sẽ được ghi
    vào environment.json và kiểm tra lại khi resume. Nếu pip báo cần restart, restart rồi chạy phần 1 lại;
    các file cache có checksum nên không ghi đè âm thầm.
    ''')
    code('''
    import importlib.metadata as metadata
    constraints = WORK / 'runtime_torch_constraints.txt'
    constraints.write_text(''.join(f'{p}=={metadata.version(p)}\\n' for p in ['torch','torchvision']))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-training-colab.txt',
                    '-c', str(constraints)], check=True)
    import torch, ultralytics
    assert torch.cuda.is_available(), 'CUDA required; stop without CPU training'
    assert ultralytics.__version__ == '8.4.178'
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(WORK/'runtime_freeze.txt').open('w'), check=True)
    ''')
    md('''
    ## 3 Prepare và full preflight

    Cấu hình mới cho phép train đúng release; release training_plan.json vẫn giữ allow_training=False.
    Source/policy/mapping/hash và 12 GiB storage reserve được kiểm tra trước train. Loader dùng bản adapter cache,
    release gốc không bị chỉnh YAML hoặc tạo label cache. Dữ liệu val không augmentation.
    ''')
    code('''
    cfg = json.loads((WORK/'configs/asl_yolo11n_auto_v2_training.json').read_text())
    cfg['runs_root'] = str(DRIVE_PROJECT/'runs'/'auto_v2')
    CONFIG = WORK/'configs/asl_yolo11n_auto_v2_colab.json'
    CONFIG.write_text(json.dumps(cfg, indent=2))
    def cli(*args):
        subprocess.run([sys.executable, '-u', 'scripts/train_release.py', '--config', str(CONFIG), *map(str,args)],
                       cwd=WORK, check=True)
    cli('preflight', '--output', WORK/'reports/asl_yolo11n_auto_v2_training_20261113/verification/colab_preflight')
    ''')
    md('''
    ## 4 Kiểm tra phần mềm và smoke 3 epoch

    Tests có fixture hình học nhân tạo, chỉ kiểm tra code. Smoke lấy subset cố định 4 train/2 val mỗi class/capture type,
    giữ split gốc, lưu danh sách ID/seed; không phải baseline full dataset. Mỗi epoch in Accuracy, macro F1 supported,
    detection metrics và commit checkpoint về Drive. `model_load.json` kiểm tra load/predict thật sau smoke.
    ''')
    code('''
    subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_training_release.py', '-q',
                    '--basetemp', '/content/pytest_asl_training'], check=True)
    START_SMOKE = False
    if START_SMOKE:
        cli('train', '--mode', 'smoke')
    ''')
    md('''
    ## 5 Kiểm tra dừng và resume GPU thật

    Run riêng tổng 4 epoch: process thứ nhất commit epoch1 rồi thoát có kiểm soát; process/model thứ hai
    khôi phục optimizer/raw model/EMA/scheduler/RNG và chạy epoch2..4. Bằng chứng kiểm tra history không trùng,
    optimizer được khôi phục và model load/predict được. Smoke3 + train mới50 không thay thế bước này.
    ''')
    code('''
    START_RESUME_PROBE = False
    if START_RESUME_PROBE:
        cli('resume-probe')
    ''')
    md('''
    ## 6 Baseline mới tối đa 50 epoch

    Chỉ được chạy khi gate smoke và resume probe thật đã đạt với cùng code/release.
    Baseline dùng đầy đủ 9042 train / 3909 val, bắt đầu lại từ pretrained COCO, không dùng weight smoke.
    SGD/lr/scheduler/augmentation và layer trainable được lưu sau resolve; native DFL cố định vẫn frozen.
    Early stopping tắt; chọn best macro F1 tại .25, tie Accuracy rồi epoch sớm hơn, tách native best.pt.
    OOM giữ run lỗi và mở run mới batch8→4→2→1 từ pretrained. Xem `baseline_estimate.json` của smoke trước khi chạy;
    ước lượng theo tỷ lệ ảnh chỉ sơ bộ, thời gian epoch đầu baseline mới đáng tin hơn.
    ''')
    code('''
    START_BASELINE = False
    if START_BASELINE:
        cli('train', '--mode', 'baseline')
    ''')
    md('''
    ## 7 Phân tích validation và xuất báo cáo

    Điền RUN rõ ràng từ output baseline, không tự chọn run mới nhất. Nếu OOM làm batch nhỏ hơn, luôn dùng config.json
    của chính run khi resume/analysis. Sweep .05..95 theo bước .05 chỉ trên val; cả model và threshold chọn trên val
    nên là kết quả phát triển, chưa độc lập. `nothing` val positive support0 nên recall/F1=N/A; background báo riêng.
    PNG300/SVG, CSV/JSON, captions, confusion NO_DETECTION, Coverage/Accuracy/Wilson/MN và gallery đều từ dự đoán thật.
    ''')
    code('''
    RUN = DRIVE_PROJECT/'runs'/'auto_v2'/'REPLACE_WITH_BASELINE_RUN_ID'
    def run_cli(*args):
        subprocess.run([sys.executable, '-u', 'scripts/train_release.py', '--config', str(RUN/'config.json'),
                        *map(str,args)], cwd=WORK, check=True)
    ANALYZE_VALIDATION = False
    if ANALYZE_VALIDATION:
        run_cli('analyze', '--run', RUN)
        from IPython.display import display, Image, Markdown
        display(Markdown((RUN/'REPORT.md').read_text()))
        display(Image(filename=str(RUN/'figures/learning_curves.png')))
        display(Image(filename=str(RUN/'figures/coverage_threshold.png')))
    ''')
    md('''
    ## 8 Resume một run bị ngắt

    Chạy lại prepare/preflight để khôi phục source/cache. Trước resume kiểm tra job GPU cũ đã ngừng; browser disconnect
    không đủ. Run hoàn tất không resume; warm start hoặc đổi cấu hình là thí nghiệm mới. `status` trả checkpoint generation
    mới nhất có checksum. Nếu last_resume alias ghi dở dùng generation đó; nếu journal hỏng dùng `recover --run RUN`
    để phục hồi prefix epoch hợp lệ, giữ artifact hỏng. Lock khác host cần tự xác minh job cũ đã ngừng rồi mới gọi
    `unlock --run RUN --old-job-stopped`; không gỡ lock chỉ vì tab mất kết nối.
    ''')
    code('''
    CHECK_RESUME_STATUS = False
    if CHECK_RESUME_STATUS:
        run_cli('status', '--run', RUN)
    # Copy exact checkpoint generation returned by status; do not guess epochs.
    CHECKPOINT = RUN/'checkpoints'/'REPLACE_WITH_VALID_GENERATION'/'resume.pt'
    START_RESUME = False
    if START_RESUME:
        run_cli('resume', '--run', RUN, '--checkpoint', CHECKPOINT)
    ''')
    md('''
    ## 9 Tái tạo hình và benchmark tùy chọn

    plots chỉ đọc artifact, không inference lại. Benchmark nếu bật: batch1,20 warmup,500 lần forward YOLO PyTorch
    float32 trên tensor GPU nạp sẵn; mean/p95 và throughput suy ra. Không gồm preprocess/NMS/MediaPipe/camera/display,
    không phải webcam FPS Windows. Không có cell mở final test. Không tắt runtime/xóa cache trước khi kiểm tra artifact Drive.
    ''')
    code('''
    REGENERATE_PLOTS = False
    RUN_BENCHMARK = False
    if REGENERATE_PLOTS:
        run_cli('plots', '--run', RUN)
    if RUN_BENCHMARK:
        run_cli('benchmark', '--run', RUN)
    ''')
    for cell in cells:
        cell['source']=cell['source'].replace('REPLACE_WITH_NEW_BUNDLE_FOLDER',args.drive_bundle_folder)
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                     'language_info':{'name':'python'},'colab':{'name':'03_asl_auto_v2_training.ipynb'}},
                  nbformat=4,nbformat_minor=5)
    for i,cell in enumerate(cells): cell['id']=f'asl_auto_v2_{i:02d}'
    out=ROOT/'notebooks/03_asl_auto_v2_training.ipynb'
    out.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__': main()
