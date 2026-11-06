"""Same pooled ordinary least squares, centered and deterministically summed."""
import math
import numpy as np

def stable_calibrate(train_predictions,train_labels):
    x=train_predictions.reshape(-1).astype(np.float64).tolist()
    y=train_labels.reshape(-1).astype(np.float64).tolist()
    if len(x)!=len(y) or not x or not all(math.isfinite(v) for v in x+y):
        raise ValueError('Invalid training-only calibration inputs')
    mx=math.fsum(x)/len(x);my=math.fsum(y)/len(y)
    xx=math.fsum((v-mx)**2 for v in x)
    if xx==0:
        # Minimum-norm OLS solution for a constant predictor, matching the model definition.
        intercept=my/(mx*mx+1);slope=mx*intercept
    else:
        slope=math.fsum((u-mx)*(v-my) for u,v in zip(x,y))/xx
        intercept=my-slope*mx
    coef=np.asarray([slope,intercept],dtype=np.float64)
    if not np.isfinite(coef).all():raise ValueError('Nonfinite calibration')
    return coef
