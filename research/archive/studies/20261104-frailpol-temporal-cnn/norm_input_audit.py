"""Outcome-blind orientation/sensor-side invariant inputs, staged only, no fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'norm-input';O=B/'norm-input'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def symmetric(a):return np.stack([(a[:,0]+a[:,2])/2,(a[:,1]+a[:,3])/2,np.abs(a[:,0]-a[:,2]),np.abs(a[:,1]-a[:,3]),a[:,4],a[:,5]],axis=1)
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);source=B/'native-motion/data.npz';d=np.load(source);raw=d['native'];valid=np.repeat(raw[:,-1:]>0,6,axis=1);cols=[3,7,11,15,19,23];norm=raw[:,cols].copy();assert len(norm)==594 and np.all(norm[~valid]==0);actual=[];rotated=[];rng=np.random.default_rng(20261105)
    for lo in [0,4,8,12,16,20]:
        a=raw[:,lo:lo+3];q,_=np.linalg.qr(rng.normal(size=(3,3)));assert np.allclose(q.T@q,np.eye(3),atol=1e-14,rtol=1e-14);actual.append(np.linalg.norm(a,axis=1));rotated.append(np.linalg.norm(np.einsum('ij,njt->nit',q,a),axis=1))
    actual=np.stack(actual,axis=1);rotated=np.stack(rotated,axis=1);assert np.allclose(actual,norm,atol=1e-10,rtol=1e-10) and np.allclose(rotated,norm,atol=1e-10,rtol=1e-10);sym=symmetric(norm);swapped=norm[:,[2,3,0,1,4,5]];assert np.array_equal(sym,symmetric(swapped));derivative=np.zeros_like(norm);dm=np.zeros_like(valid);dm[:,:,1:]=valid[:,:,1:]&valid[:,:,:-1];derivative[:,:,1:]=np.diff(norm,axis=2)*50;derivative[~dm]=0
    # A static synthetic sequence has exactly zero derivative. First sample is
    # deliberately unknown, with false derivative mask rather than a true zero.
    const=np.full((1,6,208),3.);assert np.array_equal(np.diff(const,axis=2)*50,np.zeros((1,6,207))) and not dm[:,:,0].any()
    proto=dict(code_sha256=sha(Path(__file__)),source_sha256=sha(source),people=594,design='Samefrozennativecore and50Hz samples; sixorientation-invariantAcc/Gyrnorms fromleftankle/rightankle/sacrum. No TUGlabels accessed ormodel fitted. Alsofixedleft-rightmean/absolute-difference representation loseslaterality butmayfitglobalfunctionalTUG. Preserveexplicitvalue/derivativemasks andfirstderivativeunknown. Bothvariantsretainselectedcoreonly;nooutsidegait/chair/turn/globalTUGtiming.',checks='All594storednormchannels matchXYZnorm andindependentsensororthogonalrotation;left-rightswapinvarianceforsymmetricrepresentation;staticderivativeexactzero;invalidderivativesremainfalse. These are arithmeticinvariance proofs,notpredictionaccuracy.',limits='RawIMUnotMediaPipe/RGB. Original50Hzresamplingcanaliascontentabove25Hz;notananti-aliasfiltervalidation. Absoluteleft-rightdifferencecanhavekinksandderivativesamplifynoise. Futurecandidateonly;do notchangeongoingnative-motiontraining.',expected_additional_GB=.01)
    if(R/'protocol.json').exists():assert json.loads((R/'protocol.json').read_text())==proto
    else:save(R/'protocol.json',proto)
    np.savez_compressed(O/'inputs.npz',ids=d['ids'],fold=d['fold'],F=d['F'],norm=norm,norm_valid=valid,symmetric_norm=sym,symmetric_valid=valid,norm_derivative=derivative,derivative_valid=dm)
    save(R/'verification.json',dict(status='passed_input_invariance_only_no_model_fitted',people=594,shape=list(norm.shape),original_norm_max_delta=float(np.max(np.abs(actual-norm))),rotation_max_delta=float(np.max(np.abs(rotated-norm))),left_right_swap_exact=True,first_derivative_explicitly_missing=True,output_sha256=sha(O/'inputs.npz'),protocol_sha256=sha(R/'protocol.json'),TUG_labels_accessed=False));print('594norminputinvariancepassed;no trainingstarted')
if __name__=='__main__':main()
