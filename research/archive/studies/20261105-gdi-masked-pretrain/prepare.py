"""Reuse frozen features; recover observed-coordinate masks, never protected cases."""
import csv, gzip, hashlib, json, shutil, struct, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-20261105')
OLD=ROOT/'studies/20261105-matched-normal100'; B=HERE/'bundle'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def observed(raw):
    a=raw.reshape(-1,25,3); ids=np.array([[14,13,12,19],[11,10,9,22]])
    v=np.isfinite(a).all(-1)&(a[:,:,2]>0)&(a[:,:,0]!=0)&(a[:,:,1]!=0)
    return np.repeat(v[:,ids].transpose(1,0,2),2,axis=-1)

def main():
    assert not (B/'manifest.json').exists(),'Preserve sealed experiment'
    BIG.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    oldp=read(OLD/'bundle/study-protocol.json'); parts={}
    for cohort in ['adult','child']:
        p=Path('D:/gait-research-data/D_P/matched-normal100-20261105')/cohort/'data.npz'
        assert sha(p)==oldp['data_sha256'][cohort]
        with np.load(p) as z:parts[cohort]={k:z[k].copy() for k in z.files}
    ad,ch=parts['adult'],parts['child']
    for k in ad:
        n=11261 if k in ['x','window_record','window_start','canonical_fallback'] else 1061
        assert np.array_equal(ad[k][:n],ch[k][:n]),k
    d={k:v.copy() for k,v in ad.items()}
    for k in d:
        n=11261 if k in ['x','window_record','window_start','canonical_fallback'] else 1061
        v=ch[k][n:].copy()
        if k=='window_record':v+=12
        d[k]=np.concatenate([d[k],v])
    assert d['x'].shape==(11285,2,124,12) and len(d['y'])==1085
    d['y'][1061:]=np.nan;d['label_kind']=np.array(['measured_gdi']*1061+['unlabeled']*24)
    masks=np.zeros((11285,2,124,8),bool); membership=read(OLD/'membership.json')
    receipt=[]
    # Original Stanford converter starts with file frame1; window starts are zero-based.
    archive=Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip')
    sys.path.insert(0,str(ROOT/'tools/stanford-sample-check'));import inspect_archive as ar
    ar.ARCHIVE=archive
    lookup={'openpose/'+str(v).split(':',1)[1]+'-processed':i for i,v in enumerate(d['video_id'][:1061])}
    seen=set();members={};group=None
    with archive.open('rb',buffering=4194304) as index,archive.open('rb',buffering=4194304) as stream:
        count,size,offset=ar.directory_info(index);index.seek(offset)
        def process(folder,items):
            if folder not in lookup:return
            row=lookup[folder]; wi=np.flatnonzero(d['window_record']==row)
            needed=set(int(k)+j+1 for k in d['window_start'][wi] for j in range(124))
            raw=np.full((max(needed),75),np.nan);digest=hashlib.sha256()
            for frame in sorted(needed):
                m=items[frame];data=ar.read_member(m);digest.update(data)
                candidates=[]
                for person in json.loads(data)['people']:
                    a=np.asarray(person['pose_keypoints_2d'],dtype=float)
                    if a.shape==(75,) and np.isfinite(a).all() and a[::3].max()<320:candidates.append(a)
                if len(candidates)==1:raw[frame-1]=candidates[0]
            valid=observed(raw)
            for w in wi:
                start=int(d['window_start'][w]);masks[w]=valid[:,start:start+124]
            seen.add(row);receipt.append(dict(row=row,frames=len(needed),source_content_sha256=digest.hexdigest()))
            if len(seen)%200==0:print('OBSERVATION_MASKS',len(seen),flush=True)
        for _ in range(count):
            f=struct.unpack('<4s6H3L5H2L',index.read(46));assert f[0]==b'PK\x01\x02'
            name=index.read(f[10]).decode();index.seek(f[11]+f[12],1)
            folder='/'.join(name.split('/')[:2])
            if folder!=group:
                process(group,members);group=folder;members={}
            if folder in lookup and name.endswith('_keypoints.json'):
                frame=int(name.rsplit('_',2)[-2]);members[frame]=dict(name=name,size=f[9],compressed_size=f[8],crc32=f[7],offset=f[16],method=f[4],flags=f[3])
        process(group,members);assert index.tell()==offset+size
    assert len(seen)==1061
    rows=list(csv.DictReader((ROOT/'studies/20261104-ljmu-side-audit/usable-videos.csv').open(encoding='utf-8-sig')))
    byid={r['영상ID']:r for r in rows}
    for i,r in enumerate(membership['adult']):
        p=Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104/poses')/byid[r['record']]['좌표압축파일']
        assert sha(p)==r['pose_sha256']; frames=json.load(gzip.open(p,'rt'));raw=np.full((len(frames),75),np.nan)
        for k,v in enumerate(frames):
            assert k==v['frame']
            if len(v['data']['people'])==1:raw[k]=v['data']['people'][0]['pose_keypoints_2d']
        masks[11261+i]=observed(raw)[:,r['start']:r['start']+124]
    pose=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
    for i,r in enumerate(membership['child']):
        raw=np.full((124,75),np.nan);digests=[]
        for k,selected in enumerate(r['person_indices']):
            p=pose/r['record']/('%s_%012d_keypoints.json'%(r['record'],r['start']+k));digests.append(sha(p))
            if selected>=0:raw[k]=read(p)['people'][selected]['pose_keypoints_2d']
        assert hashlib.sha256(''.join(digests).encode()).hexdigest()==r['pose_sequence_sha256']
        masks[11273+i]=observed(raw)
    d['observed_xy']=masks
    assert masks.mean(axis=(1,2,3)).min()>.5
    assert len(set(d['person_id'][:1061]))==692
    np.savez_compressed(BIG/'data.npz',**d)
    # Copy only model definitions and existing licenses, not sealed training scripts.
    shutil.copyfile(OLD/'bundle/CNN/model.py',B/'cnn.py')
    shutil.copytree(OLD/'bundle/STT/model',B/'model',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(OLD/'bundle/STT/vendor/einops',B/'einops',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
    for family in ['CNN','STT']:shutil.copyfile(OLD/'bundle'/family/'protocol.json',B/(family+'-original.json'))
    sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'));import torch
    baselines=read(ROOT/'studies/20261104-ljmu100-mixed/bundle/baseline-models.json');epochs={f:{} for f in ['CNN','STT']};proof=[]
    for item in baselines:
        if item['family'] not in epochs or item['tag']!='refit':continue
        p=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')/item['checkpoint'];assert sha(p)==item['checkpoint_sha256']
        ck=torch.load(p,map_location='cpu',weights_only=True);ident=ck['identity'];f=item['fold']
        expected=set(d['source_person_id'][:1061][d['fold'][:1061]!=f]);assert set(map(str,ident['train_people']))==expected
        epochs[item['family']][str(f)]=ident['epochs'];proof.append(dict(family=item['family'],fold=f,epochs=ident['epochs'],checkpoint_sha256=sha(p),selection='Previously frozen Stanford-only nested inner validation; no new arm outcomes used'))
    protocol=dict(version=1,study='gdi-masked-pretrain-20261105',data_sha256=sha(BIG/'data.npz'),
        conditions=['A','B_adult','B_child','C','D_adult','D_child'],families=['CNN','STT'],folds=list(range(5)),
        gdi_epochs=epochs,seed=20261028,pretrain_steps=2000,pretrain_batch_pairs=32,pretrain_lr=.0003,
        masking='Four nonoverlapping8-frame spans, one random position within each31-frame quarter.32/124 frames; same both sides; all input features hidden. Learned mask token.',
        target='Existing smoothed/normalized 2D xy, only original observed finite nonzero points with confidence>0; not anatomical ground truth.',
        pretrained_keys={'CNN':['convs.','norms.'],'STT':['joint_embedding.','spatial_positional_enc','temporal_positional_enc','spatial_block.','temporal_block.','spatial_norm.','temporal_norm.']},
        training='Fresh models. Same initial GDI head, seed, per-fold Stanford-derived epochs, batch size and exact GDI example/update counts across A-D within each family. C/D add equal2000 pretraining updates. A/B have no pretraining; total two-stage compute is not equal.',
        supervised_sampling='A/C/D: identical shuffled Stanford windows/sides and original equal-record loss. B: shuffle Stanford+matched external window-sides, take exactly Stanford sample count without replacement; original record weights, external total1 per person.',
        pretrain_sampling='Uniform person then record then window. D replaces same Stanford draws by external draws at probability9/(Stanford training people+9); identical external flags/counts across adult/child. C has no external. No GDI labels passed to pretraining.',
        model_selection='No arm-specific selection. Reuse already selected baseline inner-CV epoch per outer fold for every arm. Each outer fold starts independently; no pretraining on held-out Stanford coordinates. Final223 unopened.',
        calibration='Stanford training rows only; same shared OLS and target min/max. Final unmasked Stanford training predictions determine calibration.',
        evaluation='Same692 Stanford people,1061 records. all, actual record-side>=90,>=100,<70. Equal-person MAE, mean(max(y-p,0)), signed bias, underprediction rate. Paired10000 person bootstrap. No normal100 MAE endpoint.',
        expected_gdi_fits=60,expected_pretraining_fits=30,cohorts={'adult':{'people':9,'records':12},'child':{'people':9,'records':12}},
        limits='Fixed single seed schedule, single normal sample; source differences remain. No final-test or clinical validation. Matched training budget comparison rather than per-arm optimum hyperparameters.')
    save(B/'protocol.json',protocol);save(HERE/'epoch-provenance.json',proof)
    save(HERE/'preparation.json',dict(passed=True,data_sha256=protocol['data_sha256'],archive_bytes=archive.stat().st_size,observed_fraction=float(masks.mean()),minimum_window_observed_fraction=float(masks.mean(axis=(1,2,3)).min()),shape=list(d['x'].shape),normal_labels_absent=bool(np.isnan(d['y'][1061:]).all()),source_membership_sha256=sha(OLD/'membership.json'),gdi_epochs=epochs,stanford_raw_receipts=receipt))
    print(json.dumps({k:v for k,v in read(HERE/'preparation.json').items() if k!='stanford_raw_receipts'},indent=2))

if __name__=='__main__':main()
