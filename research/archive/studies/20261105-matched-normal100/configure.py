"""Create isolated training code; sealed prior studies remain unchanged."""
import shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261105-child100-train-only';B=HERE/'bundle'
assert not (B/'manifest.json').exists(),'Preserve sealed training code; use a new study revision'
B.mkdir(exist_ok=True)
def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)
for family in ['CNN','STT','bilateral']:
    shutil.copytree(OLD/'bundle'/family,B/family,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    p=B/family/'windows.py';s=p.read_text()
    s=replace(s,'starts=None,fallback=None):','starts=None,fallback=None,record_weight=None):')
    s=replace(s,'self.nrecords=int(nrecords)','self.nrecords=int(nrecords)\n        self.record_weight=np.ones(nrecords) if record_weight is None else np.asarray(record_weight,dtype=float)\n        assert self.record_weight.shape==(nrecords,) and np.isfinite(self.record_weight).all() and (self.record_weight>0).all()')
    s=replace(s,'self.starts[keep],self.fallback[keep])','self.starts[keep],self.fallback[keep],self.record_weight[indices])')
    s=replace(s,'weight=len(self.owner)/(self.nrecords*self.counts[self.owner])','weight=len(self.owner)*self.record_weight[self.owner]/(self.record_weight.sum()*self.counts[self.owner])')
    p.write_text(s,encoding='utf-8')
    p=B/family/'run.py';s=p.read_text()
    if family=='CNN':
        s=replace(s,'lo, span = scale_from_train(y)','calibration_mask=np.array([str(p).startswith("Stanford:") for p in people])\n    require(calibration_mask.any(),"No Stanford calibration rows")\n    lo, span = scale_from_train(y[calibration_mask])')
        s=replace(s,'correction=calibrate(train_raw,y)','correction=calibrate(train_raw[calibration_mask],y[calibration_mask])')
    else:
        s=replace(s,"lo=float(np.float32(y.min()));span=float(np.float32(max(float(y.max()-y.min()),1e-6)))",'calibration_mask=np.array([str(p).startswith("Stanford:") for p in ids])\n    require(calibration_mask.any(),"No Stanford calibration rows")\n    cy=y[calibration_mask]\n    lo=float(np.float32(cy.min()));span=float(np.float32(max(float(cy.max()-cy.min()),1e-6)))')
        s=replace(s,'coef=calibration(raw,y)','coef=calibration(raw[calibration_mask],y[calibration_mask])')
    p.write_text(s,encoding='utf-8')
for name in ['ensemble.json','baseline-models.json']:shutil.copyfile(OLD/'bundle'/name,B/name)
s=(OLD/'bundle/experiment.py').read_text()
s=replace(s,"ap.add_argument('--family'","ap.add_argument('--arm',choices=['adult','child'],required=True)\nap.add_argument('--family'")
s=replace(s,"protocol['data_sha256']","protocol['data_sha256'][a.arm]")
s=replace(s,"d.pop('canonical_fallback'))","d.pop('canonical_fallback'),d['record_weight'])")
s=replace(s,"assert len(d['y'])==1071 and len(set(d['person_id']))==699","assert len(d['y'])==protocol['counts'][a.arm]['records']\n    assert len(set(d['person_id']))==692+protocol['matched_people']")
s=replace(s,"d['source']=='LIVE-TD'","d['source']!='Stanford'")
s=replace(s,"child.sum()==10","child.sum()==protocol['counts'][a.arm]['extra_records']")
s=replace(s,"'arm':'Stanford_plus_child100_training_only'","'arm':a.arm")
s=replace(s,"np.std(train_raw,axis=0)","np.std(train_raw[d['source'][tr]=='Stanford'],axis=0)")
s=replace(s,"'synthetic%d'%i","'Stanford:synthetic%d'%i")
(B/'experiment.py').write_text(s,encoding='utf-8')
# RF retains its original uncalibrated output; only external sample weights change.
s=(OLD/'train_rf.py').read_text()
s=replace(s,'import hashlib,json,sys,time','import argparse,hashlib,json,sys,time')
s=replace(s,"BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')","ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['adult','child'],required=True);a=ap.parse_args()\nBIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')/a.arm")
s=replace(s,"protocol=read(HERE/'rf-protocol.json');fingerprint=sha(HERE/'rf-protocol.json')","protocol=read(HERE/'rf-protocol.json');fingerprint=sha(HERE/'rf-protocol.json')")
s=replace(s,"protocol['data_sha256']","protocol['data_sha256'][a.arm]")
s=replace(s,"m.fit(x[ix].reshape(-1,48),y[ix].reshape(-1))","m.fit(x[ix].reshape(-1,48),y[ix].reshape(-1),sample_weight=np.repeat(d['record_weight'][ix],2))")
s=replace(s,"d['source']=='LIVE-TD'","d['source']!='Stanford'")
s=replace(s,"HERE/'rf-run-summary.json'","HERE/('rf-run-summary-'+a.arm+'.json')")
(HERE/'train_rf.py').write_text(s,encoding='utf-8')
print('Isolated code generated; no models trained.')
