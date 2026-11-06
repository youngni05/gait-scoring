"""Select only from held-out INNER predictions, no outer outcomes accepted."""
import numpy as np

def person_mae(y,p,ids):
    y=np.asarray(y);p=np.asarray(p);ids=np.asarray(ids)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or len(ids)!=len(y):raise ValueError('Axes mismatch')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
    errors=np.abs(y-p).mean(1)
    return float(np.mean([errors[ids==i].mean() for i in np.unique(ids)]))

def choose_weight(inner_y,inner_base,inner_rank,inner_people,grid):
    if np.shape(inner_base)!=np.shape(inner_rank):raise ValueError('Mismatched models')
    grid=sorted(set(float(w) for w in grid))
    if not grid or grid[0]<0 or grid[-1]>1:raise ValueError('Invalid convex weights')
    scores=[person_mae(inner_y,(1-w)*inner_base+w*inner_rank,inner_people) for w in grid]
    best=min(scores)
    selected=next(i for i,s in enumerate(scores) if s<=best+1e-12)
    return dict(rank_weight=grid[selected],inner_person_mae=scores[selected],grid=grid,scores=scores)
