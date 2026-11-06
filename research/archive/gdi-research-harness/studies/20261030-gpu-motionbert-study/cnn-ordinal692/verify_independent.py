"""Small independent formulas for rank probabilities and person metrics."""
import math,json
import numpy as np
import torch
import run as r
from ordinal import probabilities,targets
r.setup(1,'cpu')
logits=torch.tensor([[[1.4,.2,-1.7]]],dtype=torch.float64)
p=probabilities(logits).numpy()[0,0]
q=[1/(1+math.exp(-v)) for v in [1.4,.2,-1.7]]
assert np.max(abs(p-np.array([1-q[0],q[0]-q[1],q[1]-q[2],q[2]])))<1e-15
y=np.array([[69.9,70],[81,79],[39,101]],float);ids=np.array(['a','a','b']);raw=y+np.array([[1,-2],[3,-4],[5,-6]])
pr=np.full((3,2,18),.5/17);cls=targets(y,10)
for i in range(3):
    for side in range(2):pr[i,side,cls[i,side]]=.5
got=r.metrics(y,raw,pr,ids,10)
assert abs(got['mae']-4.)<1e-12
assert abs(got['nll']-math.log(2))<1e-12
assert abs(got['brier']-(.25+17*(.5/17)**2))<1e-12 and got['accuracy']==1
assert cls.tolist()==[[7,8],[9,8],[4,11]]
result=dict(passed=True,probability_difference_formula=True,person_mean_mae=True,independent_log_loss=True,independent_brier=True,half_open_edges=True)
r.save(r.H/'checks/independent_formulas.json',result);print(json.dumps(result))
