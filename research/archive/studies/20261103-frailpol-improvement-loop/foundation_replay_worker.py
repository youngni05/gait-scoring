"""One original foundation context per process; original numerical init order."""
import sys,json
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
from round3 import predict2,np,joblib
import torch
assert torch.get_num_threads()==4
path,request,output,wrapped=sys.argv[1:]
model=joblib.load(path)
if wrapped=='1':model=model['base']
prediction=predict2(model,np.load(request))
assert np.isfinite(prediction).all()
np.save(output,prediction)
print(json.dumps(dict(status='predicted',threads=torch.get_num_threads(),people=len(prediction))))
