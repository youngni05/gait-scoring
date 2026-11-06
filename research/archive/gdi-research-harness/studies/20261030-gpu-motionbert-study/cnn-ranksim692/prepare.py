"""One authorized follow-up: same person sampler, MSE vs MSE+RankSim."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;S=H.parent/'cnn-person-bmc692'
for n in ['run.py','model.py','windows.py','balanced.py','audit.py','review.py','verify.py','worker.py','reference_metrics.json','input_contract.json','seal.py','deploy.py','receive_pilot.py']:
    s=(S/n).read_text(encoding='utf-8')
    if n not in ['balanced.py','model.py','windows.py','audit.py','reference_metrics.json','input_contract.json']:
        s=s.replace("['mse','bmc']","['mse','ranksim']").replace("['mse', 'bmc']","['mse', 'ranksim']")
        s=s.replace('cnn-person-bmc692','cnn-ranksim692').replace('gdi-cnn-person-bmc692-20261031','gdi-cnn-ranksim692-20261101').replace('gdi-cnn-ranksim692-20261031','gdi-cnn-ranksim692-20261101')
        s=s.replace('gdi-person-bmc-','gdi-ranksim-')
    if n=='run.py':
        s=s.replace('from balanced import sample_epoch','from rank_aux import feature_loss\nfrom balanced import sample_epoch')
        s=s.replace("pred=model(inputs)[:,0]","outputs,features=model(inputs,return_features=True);pred=outputs[:,0]")
        s=s.replace("data_loss=objective(pred,targets,c['loss_variant'],c['sigma_gdi']/float(span))","data_loss=(pred-targets).square().mean()\n            if c['loss_variant']=='ranksim':data_loss=data_loss+c['rank_weight']*feature_loss(features,targets,c['rank_interpolation'])")
    if n=='model.py':
        s=s.replace('def forward(self, x):\n        if x.ndim', 'def forward(self, x, return_features=False):\n        if x.ndim')
        s=s.replace('return self.output(F.relu(self.hidden(x)))','features=F.relu(self.hidden(x))\n        output=self.output(features)\n        return (output,features) if return_features else output')
    if n=='seal.py':s=s.replace("'input_contract.json']","'input_contract.json','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE']")
    if n=='verify.py':
        s=s.replace('from balanced import objective,','from rank_aux import feature_loss\nfrom balanced import objective,')
        s=s.replace("    # Real CNN on synthetic windows", "    features=torch.randn(8,10,device=a.device,requires_grad=True)\n    target=torch.tensor([1.,1.,2.,3.,4.,5.,6.,7.],device=a.device)\n    aux=feature_loss(features,target,2.0);aux.backward()\n    assert torch.isfinite(aux) and torch.isfinite(features.grad).all()\n    # Real CNN on synthetic windows")
    if n=='deploy.py':s=s.replace("'checks/cpu-v2/verification.json'","'checks/cpu-v1/verification.json'")
    (H/n).write_text(s,encoding='utf-8')
c=json.loads((S/'protocol.json').read_text());c.update(name='cnn_person_mse_ranksim',variants=['mse','ranksim'],rank_weight=1.0,rank_interpolation=2.0,loss='MSE vs MSE+mean official RankSim regularizer',note='One follow-up. Same sampler, same initialization and OLS. Fixed rank weight1 and interpolation2; no outer-selected tuning.')
(H/'protocol.json').write_text(json.dumps(c,indent=2)+'\n')
print('Prepared isolated RankSim comparison')
