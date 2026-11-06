"""Exactly the previous GDI predictor with only isolated input/output paths."""
from common import *
if __name__=='__main__':
    worker=module('original_frozen_predict',OLD/'predict.py');worker.HERE=HERE;worker.BIG=BIG;worker.main()
