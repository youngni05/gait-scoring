"""Invoke the previous frozen prediction code with isolated input/output paths."""
from common import *

if __name__ == '__main__':
    old = module('frozen_gdi_predict', OLD/'predict.py')
    old.HERE = HERE
    old.BIG = BIG
    old.main()
