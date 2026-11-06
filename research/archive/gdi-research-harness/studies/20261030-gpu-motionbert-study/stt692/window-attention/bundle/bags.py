"""Gather all windows of each recording; record IDs determine split membership."""
import numpy as np
import torch

def pack(x,indices,device,dtype=torch.float32):
    indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1 or not len(indices) or len(np.unique(indices))!=len(indices) or (indices<0).any() or (indices>=len(x)).any():
        raise ValueError('Invalid recording indices')
    rows=[np.flatnonzero(x.owner==i) for i in indices]
    if any(not len(r) for r in rows):raise ValueError('Empty recording')
    width=max(map(len,rows));values=np.zeros((len(indices),width,2,124,4,2),dtype=np.float32)
    valid=np.zeros((len(indices),width),dtype=bool);window_index=np.full(valid.shape,-1,dtype=np.int64)
    for k,ix in enumerate(rows):
        values[k,:len(ix)]=x.values[ix];valid[k,:len(ix)]=True;window_index[k,:len(ix)]=ix
    return torch.as_tensor(values,device=device,dtype=dtype),torch.as_tensor(valid,device=device),window_index

def predict_details(model,x,lo,span,device,batch_records=8):
    model.eval();dtype=next(model.parameters()).dtype
    predictions=np.zeros((len(x),2),dtype=np.float64)
    weights=np.zeros((len(x.values),2),dtype=np.float64);scores=np.zeros_like(weights);seen=np.zeros(len(weights),int)
    with torch.inference_mode():
        for start in range(0,len(x),batch_records):
            ix=np.arange(start,min(start+batch_records,len(x)))
            values,mask,mapping=pack(x,ix,device,dtype)
            p,w,s=model(values,mask,details=True);valid=mapping>=0;flat=mapping[valid]
            predictions[ix]=p.cpu().numpy().astype(np.float64)*span+lo
            weights[flat]=w.cpu().numpy()[valid];scores[flat]=s.cpu().numpy()[valid].astype(np.float64)*span+lo;seen[flat]+=1
    if not (seen==1).all():raise ValueError('Window assignment not one-to-one')
    sums=np.zeros((len(x),2));np.add.at(sums,x.owner,weights)
    np.testing.assert_allclose(sums,1,rtol=0,atol=2e-7)
    if not np.isfinite(predictions).all() or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid pooling output')
    return predictions,dict(weights=weights,window_scores=scores,window_record=x.owner.copy(),window_start=x.starts.copy(),canonical_fallback=x.fallback.copy())

def weight_summary(x,details):
    w=details['weights'];entropy=np.zeros((len(x),2));squares=np.zeros_like(entropy);maximum=np.zeros_like(entropy)
    np.add.at(entropy,x.owner,-w*np.log(np.maximum(w,1e-300)));np.add.at(squares,x.owner,w*w);np.maximum.at(maximum,x.owner,w)
    return dict(mean_effective_windows=float(np.exp(entropy).mean()),mean_inverse_squared_weight=float((1/squares).mean()),mean_max_weight=float(maximum.mean()),single_window_records=int((x.counts==1).sum()),normalization='sum over windows is one separately for each record and leg',interpretation='Predictive weights, not validated visibility, causality or clinical importance')

def audit_weights(x,detail,raw,mode):
    for key,expected in [('window_record',x.owner),('window_start',x.starts),('canonical_fallback',x.fallback)]:np.testing.assert_array_equal(detail[key],expected)
    w=detail['weights'];s=detail['window_scores']
    if w.shape!=s.shape or w.shape!=(len(x.values),2) or not np.isfinite(w).all() or not np.isfinite(s).all() or (w<0).any():raise ValueError('Invalid saved window scores/weights')
    sums=np.zeros((len(x),2));pred=np.zeros_like(sums)
    np.add.at(sums,x.owner,w);np.add.at(pred,x.owner,w*s)
    np.testing.assert_allclose(sums,1,rtol=0,atol=2e-7)
    np.testing.assert_allclose(pred,raw,rtol=0,atol=1e-4)
    if mode=='mean':np.testing.assert_allclose(w,np.repeat((1/x.counts[x.owner])[:,None],2,axis=1),rtol=0,atol=1e-7)
