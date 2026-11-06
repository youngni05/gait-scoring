"""Build frozen test inputs. No model output is read here."""
import csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');BIG.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'tmp/ljmu-analysis-python-deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
from preprocess import extract_windows
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
LEG={'L':[13,14,19,20,21],'R':[10,11,22,23,24]};HIP={'L':12,'R':9}
def perturb(raw,mode):
    out=raw.copy();k=raw.reshape(-1,25,3);o=out.reshape(-1,25,3)
    for side,gain in ([('L',0.)] if mode=='left_leg_stiff' else [('L',.5),('R',.5)]):
        h=HIP[side];hv=np.isfinite(k[:,h,0])&(k[:,h,2]>0)
        for j in LEG[side]:
            jv=hv&np.isfinite(k[:,j,0])&(k[:,j,2]>0)
            if not jv.any():continue
            off=k[jv,j,:2]-k[jv,h,:2];mean=off.mean(0)
            o[jv,j,:2]=k[jv,h,:2]+mean+gain*(off-mean)
    return o.reshape(-1,75)
sets={}
child=np.load('D:/gait-research-data/D_P/matched-normal100-20261105/child/data.npz',allow_pickle=False)
live=np.flatnonzero(child['source']=='LIVE-TD');assert len(live)==12
lp=child['person_id'][live];assert len(set(lp))==9
keep=np.isin(child['window_record'],live);rec={int(r):i for i,r in enumerate(live)}
sets['live']=dict(x=child['x'][keep],owner=np.array([rec[int(r)] for r in child['window_record'][keep]]),n=12)
TEST=ROOT/'gdi-research-harness/studies/20261101-self-video-test01'
w=read(TEST/'windows.json');sx=np.load('D:/gait-research-data/D_P/self-video-test01-20261101/features.npz')['x'];assert len(w)==len(sx)
numbers=sorted({v['number'] for v in w});assert len(numbers)==13
lab=read(TEST/'labels.json')
sets['self']=dict(x=sx,owner=np.array([numbers.index(v['number']) for v in w]),n=13)
d=np.load('D:/gait-research-data/D_P/ljmu100-mixed-20261104/data.npz',allow_pickle=False)
s=d['source'][d['window_record']]=='LJMU';assert s.sum()==3305
base=d['x'][s];owner=d['window_record'][s]-1061;starts=d['window_start'][s]
rng=np.random.default_rng(20261105)
perm=np.stack([rng.permutation(124) for _ in range(len(base))])
shuffled=np.take_along_axis(base,perm[:,None,:,None],axis=2)
static=np.broadcast_to(base.mean(axis=2,keepdims=True),base.shape).copy()
sets['ljmu_original']=dict(x=base,owner=owner,n=881)
sets['ljmu_frame_shuffle']=dict(x=shuffled,owner=owner,n=881)
sets['ljmu_static_mean']=dict(x=static,owner=owner,n=881)
rows=list(csv.DictReader((ROOT/'studies/20261104-ljmu-side-audit/usable-videos.csv').open(encoding='utf-8-sig')));assert len(rows)==881
out={m:[] for m in ['left_leg_stiff','both_amp50']}
for i,r in enumerate(rows):
    frames=json.load(gzip.open(Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104/poses')/r['좌표압축파일'],'rt'));raw=np.full((len(frames),75),np.nan)
    for k,f in enumerate(frames):
        pp=f['data']['people']
        if len(pp)==1:raw[k]=pp[0]['pose_keypoints_2d']
    for m in out:
        e=extract_windows(perturb(raw,m))
        assert e['window_start'].tolist()==starts[owner==i].tolist(),(i,m)
        out[m].append(e['x'])
    if (i+1)%200==0:print(i+1,flush=True)
for m,v in out.items():sets['ljmu_'+m]=dict(x=np.concatenate(v),owner=owner,n=881)
save={}
for k,v in sets.items():
    assert v['x'].shape[1:]==(2,124,12) and np.isfinite(v['x']).all(),k
    save[k+'__x']=v['x'].astype(np.float32);save[k+'__owner']=v['owner'];save[k+'__n']=np.array(v['n'])
np.savez_compressed(BIG/'inputs.npz',**save)
meta=dict(protocol_sha256=sha(HERE/'protocol.json'),inputs_sha256=sha(BIG/'inputs.npz'),
    live_people=lp.tolist(),live_ages=child['age'][live].tolist(),live_video=child['video_id'][live].tolist(),
    self_numbers=numbers,self_normal=lab['normal_numbers'],self_abnormal=lab['abnormal_numbers'],
    ljmu_person=d['person_id'][1061:].tolist(),ljmu_fold=d['fold'][1061:].tolist(),
    window_counts={k:len(v['x']) for k,v in sets.items()})
(HERE/'preparation.json').write_text(json.dumps(meta,indent=1,ensure_ascii=False),encoding='utf-8')
print('prepared',meta['window_counts'],flush=True)
