"""Print startup stages and periodic stacks without changing replay criteria."""
import faulthandler
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
print('1/4 importing numpy', flush=True)
import numpy
print('2/4 importing torch', flush=True)
import torch
print('3/4 importing run', flush=True)
import run
print('4/4 running original check', flush=True)
sys.argv = ['run.py', 'check', '--device', 'cuda', '--output', 'checks/gpu_replay']
run.main()
faulthandler.cancel_dump_traceback_later()
print('CHECK_FINISHED', flush=True)
