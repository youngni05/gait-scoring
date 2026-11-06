"""Portable inventory and optional data-free GPU forward/backward check.

No installs, downloads, research-data reads, model training, or CPU fallback.
Exit 0: requested check completed; inventory_only is NOT a GPU success.
Exit 2: GPU/framework unavailable; exit 3: requested tensor check failed.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def inspect(tensor_check=False):
    result={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.executable,'python_version':platform.python_version(),
            'platform':platform.platform(),'status':'inventory_only',
            'tensor_check':'not_requested','packages':{},'nvidia_smi':None,
            'backend_scope':'PyTorch CUDA/ROCm and MPS; other accelerator backends are not tested by this script.',
            'limits':'This is an environment check, not a gait-model test or a model VRAM/time estimate.'}
    for name in ['torch','torchvision','numpy','mediapipe']:
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    command=shutil.which('nvidia-smi')
    if command:
        try:
            r=subprocess.run([command,'--query-gpu=name,driver_version,memory.total,memory.free',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
            result['nvidia_smi']={'returncode':r.returncode,'output':r.stdout.strip(),'error':r.stderr.strip()}
        except subprocess.TimeoutExpired:result['nvidia_smi']={'error':'query_timeout'}
    if not tensor_check:return result,0
    if not result['packages']['torch']:
        result.update(status='blocked_missing_torch',tensor_check='not_run')
        return result,2
    try:
        import torch
        result['torch_runtime']={'version':torch.__version__,'cuda_build':torch.version.cuda,
                                 'rocm_build':getattr(torch.version,'hip',None)}
        if torch.cuda.is_available():
            device=torch.device('cuda:0');properties=torch.cuda.get_device_properties(device)
            result['gpu']={'name':properties.name,'total_memory_bytes':properties.total_memory,
                           'backend':'rocm' if getattr(torch.version,'hip',None) else 'cuda'}
            torch.cuda.reset_peak_memory_stats(device)
        elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available():
            device=torch.device('mps');result['gpu']={'backend':'mps','total_memory_bytes':None}
        else:
            result.update(status='blocked_no_compatible_gpu',tensor_check='not_run')
            return result,2
        started=time.perf_counter()
        x=torch.linspace(-1,1,128*128,device=device,dtype=torch.float32).reshape(128,128).requires_grad_()
        weight=torch.eye(128,device=device,dtype=torch.float32,requires_grad=True)
        loss=(x@weight).square().mean();loss.backward()
        expected=2*x.detach()/x.numel()
        finite=bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
                    and torch.isfinite(weight.grad).all().item())
        max_error=float((x.grad-expected).abs().max().item())
        if device.type=='cuda':torch.cuda.synchronize(device)
        elif device.type=='mps':torch.mps.synchronize()
        result['synthetic_check']={'device':str(device),'finite_forward_and_gradients':finite,
              'gradient_max_absolute_error':max_error,'elapsed_seconds':time.perf_counter()-started,
              'kind':'128x128 identity-matmul analytic-gradient check; not a trained model'}
        if device.type=='cuda':result['synthetic_check']['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        if not finite or max_error>1e-6:
            result.update(status='failed_tensor_check',tensor_check='failed');return result,3
        result.update(status='gpu_tensor_check_passed',tensor_check='passed');return result,0
    except Exception as error:
        result.update(status='failed_tensor_check',tensor_check='failed',
                      error={'type':type(error).__name__,'message':str(error)})
        return result,3


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tensor-check',action='store_true',help='Explicitly request a tiny GPU forward/backward check.')
    args=parser.parse_args()
    if args.out.exists():parser.error('Output already exists; choose a new filename to preserve earlier evidence.')
    result,code=inspect(args.tensor_check)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'report':str(args.out),'exit_code':code}))
    sys.exit(code)
