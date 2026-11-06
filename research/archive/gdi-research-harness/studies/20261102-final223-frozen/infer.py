"""One frozen inference pass; outcomes are deliberately not loaded here."""
from common import *
import time
def main():
    locked();assert not (HERE/'evaluation-started.json').exists()
    audit=read(HERE/'input-audit.json');assert sha(BIG/'features.npz')==audit['features_sha256']
    with np.load(BIG/'features.npz') as z:x=z['x'];owner=z['window_record'];ids=z['person_id'];videos=z['video_id']
    with np.load(Path('D:/gait-research-data/D_P/self-video-gdi-20261101/features.npz')) as z:replay_x=z['x'][:3]
    expected=np.load('D:/gait-research-data/D_P/self-video-gdi-20261101/predictions.npz')
    manifest=read(HERE/'models.json');out={};checks=[];start=time.monotonic()
    for i,m in enumerate(manifest['members']):
        family=m['family'];f=m['fold'];tag=m.get('tag');key='bilateral_%d'%f if family=='bilateral' else '%s_%d_%s'%(family,f,tag)
        path=BIG/(key+'.npz');proofpath=path.with_suffix('.json')
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        if proofpath.exists():
            proof=read(proofpath);assert proof['protocol_sha256']==sha(HERE/'protocol.json') and proof['checkpoint_sha256']==m['checkpoint_sha256'] and proof['prediction_sha256']==sha(path) and proof['features_sha256']==audit['features_sha256']
            with np.load(path) as z:out[key]=z['prediction']
        else:
            ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True);model=prior.new_model(family);model.load_state_dict(ck['model'],strict=True)
            a,b=m['calibration']
            replay=prior.forward(model,family,replay_x,ck['lo'],ck['span'])*a+b
            error=float(np.abs(replay-expected[key][:3]).max());assert error<1e-4,(key,error)
            pred=prior.forward(model,family,x,ck['lo'],ck['span'])*a+b
            out[key]=aggregate(pred,owner,len(ids));assert np.isfinite(out[key]).all()
            np.savez_compressed(path,prediction=out[key])
            proof=dict(key=key,checkpoint_sha256=m['checkpoint_sha256'],prediction_sha256=sha(path),features_sha256=audit['features_sha256'],protocol_sha256=sha(HERE/'protocol.json'),prior_replay_max=error,optimizer_steps=0)
            save(proofpath,proof)
        checks.append(proof)
        print(json.dumps({'member':i+1,'total':len(manifest['members']),'key':key,'seconds':round(time.monotonic()-start,1)}),flush=True)
    for f,sf in enumerate(manifest['selections']):
        for family in ['CNN','STT']:
            keys=['%s_%d_%s'%(family,f,m['tag']) for m in manifest['members'] if m['family']==family and m['fold']==f]
            out['%s_%d_bag'%(family,f)]=np.mean([out[k] for k in keys],axis=0)
        out['fusion_%d'%f]=combine(out['CNN_%d_bag'%f],out['STT_%d_bag'%f],out['bilateral_%d'%f],sf['cnn_weight'],sf['fraction'])
    for name,pattern in [('CNN_refit','CNN_%d_refit'),('CNN_bag','CNN_%d_bag'),('fusion','fusion_%d')]:out[name]=np.mean([out[pattern%f] for f in range(5)],axis=0)
    np.savez_compressed(BIG/'predictions.npz',person_id=ids,video_id=videos,**out)
    expected.close()
    lock=dict(protocol_sha256=sha(HERE/'protocol.json'),models_sha256=sha(HERE/'models.json'),features_sha256=sha(BIG/'features.npz'),predictions_sha256=sha(BIG/'predictions.npz'),records=len(ids),people=len(np.unique(ids)),members=checks,labels_opened=False,optimizer_steps=0)
    save(HERE/'predictions-locked.json',lock);print('FINAL223_PREDICTIONS_LOCKED',flush=True)
if __name__=='__main__':main()
