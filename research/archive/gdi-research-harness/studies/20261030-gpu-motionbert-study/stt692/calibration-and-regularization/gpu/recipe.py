"""Factorial training recipes. Calibration remains the original record-level OLS."""
import numpy as np
import torch
VARIANTS=('baseline','person','decay','person_decay')

def configure(c,variant):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    return dict(c,name='bilateral_regularization_'+variant,variant=variant,
                person_weighted=variant in ('person','person_decay'),
                optimizer='AdamW' if variant in ('decay','person_decay') else 'Adam',
                weight_decay=0.01 if variant in ('decay','person_decay') else 0.)

def recipe_identity(c):
    return {k:c[k] for k in ('variant','person_weighted','optimizer','weight_decay','learning_rate','batch_size')}

def training_weights(x,ids,c):
    if not c['person_weighted']:return x.side_weights()
    if len(ids)!=len(x):raise ValueError('Record/person mismatch')
    people,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
    # Each person -> records -> windows -> two sides. Global mean is one.
    w=len(x.values)/(len(people)*counts[inverse[x.owner]]*x.counts[x.owner])
    return np.repeat(w,2).astype(np.float32)

def make_optimizer(model,c):
    cls=torch.optim.AdamW if c['optimizer']=='AdamW' else torch.optim.Adam
    return cls(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
