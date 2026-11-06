"""Allocation-bound CUDA and synthetic model check; no research training data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, socket, time, subprocess, importlib.util, hashlib, faulthandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
report = {'passed': False, 'job_id': os.environ.get('SLURM_JOB_ID'),
          'hostname': socket.gethostname(), 'real_data_optimizer_steps': 0}
assert report['job_id'] and report['hostname'] == 'DIS03'
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

try:
    report['interpreters'] = {}
    for exe in ('python3', '/usr/bin/python3.8'):
        code = "import sys,importlib.util,json; print(json.dumps(dict(executable=sys.executable,version=sys.version,torch_found=importlib.util.find_spec('torch') is not None)))"
        p = subprocess.run([exe, '-B', '-c', code], capture_output=True, text=True, timeout=30)
        report['interpreters'][exe] = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    print(json.dumps(report['interpreters']), flush=True)
    import numpy as np
    import torch
    report['environment'] = {'python': sys.version, 'executable': sys.executable,
        'torch': str(torch.__version__), 'torch_path': torch.__file__,
        'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    print('CUDA allocation and arithmetic', flush=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype in (torch.float32, torch.float64):
        x = torch.ones((32, 32), dtype=dtype, device='cuda')
        y = x @ x
        torch.cuda.synchronize()
        assert torch.equal(y, torch.full_like(y, 32))
    report['gpu'] = torch.cuda.get_device_name(0)
    report['vram_bytes'] = torch.cuda.get_device_properties(0).total_memory
    cnn = Path('/abr/RESEARCH_USER/gdi-cnn-record-loss692-20261031')
    stt = Path('/abr/RESEARCH_USER/gdi-stt692-bilateral-20261031')
    paths = [cnn/'model.py', cnn/'data.npz', stt/'bilateral.py', stt/'data.npz']
    report['source_sha256_before'] = {str(p): sha(p) for p in paths}
    spec = importlib.util.spec_from_file_location('diagnostic_cnn_model', cnn/'model.py')
    cnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_module)
    GdiCNN = cnn_module.GdiCNN
    sys.path.insert(0, str(stt))
    sys.path.insert(0, str(stt/'vendor'))
    from bilateral import BilateralSTT
    report['models'] = {}
    for name, factory, shape in [('CNN', GdiCNN, (32,12,124)),
                                  ('bilateral_STT', BilateralSTT, (8,2,124,4,2))]:
        print('Synthetic training:', name, flush=True)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model = factory().cuda()
        before = {k: v.detach().clone() for k,v in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(shape, device='cuda')
        losses = []
        for _ in range(3):
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred - 0.25).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            losses.append(loss.item())
        changed = sum(not torch.equal(before[k],v) for k,v in model.named_parameters())
        assert changed > 0
        model.eval()
        with torch.no_grad():
            expected = model(x)
        checkpoint = ROOT/(name+'.pt')
        torch.save(model.state_dict(), checkpoint)
        restored = factory().cuda()
        restored.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
        restored.eval()
        with torch.no_grad():
            replay = restored(x)
            cpu = restored.cpu()(x.cpu()).cuda()
        gpu_error = (replay-expected).abs().max().item()
        cpu_error = (cpu-expected).abs().max().item()
        assert gpu_error == 0 and cpu_error <= 0.0001, (gpu_error,cpu_error)
        torch.cuda.synchronize()
        report['models'][name] = {'synthetic_steps': 3, 'losses': losses,
            'changed_parameter_tensors': changed, 'gpu_reload_max_error': gpu_error,
            'cpu_replay_max_error': cpu_error, 'tolerance': 0.0001,
            'seconds': time.monotonic()-start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'checkpoint_sha256': sha(checkpoint)}
    report['source_sha256_after'] = {str(p): sha(p) for p in paths}
    assert report['source_sha256_before'] == report['source_sha256_after']
    report['passed'] = True
    print('DIS03_ENVIRONMENT_CHECK_PASSED', flush=True)
except BaseException as exc:
    report['error'] = repr(exc)
    raise
finally:
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()
