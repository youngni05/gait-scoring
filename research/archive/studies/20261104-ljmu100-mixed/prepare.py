"""검수한 LJMU 881편을 기존 스탠퍼드 입력에 연결한다. 기존 자료는 수정하지 않는다."""
import csv,gzip,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
SOURCE=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
AUDIT=ROOT/'studies/20261104-ljmu-side-audit'
BUNDLE=HERE/'bundle'
sys.path[:0]=[str(ROOT/'tmp/ljmu-analysis-python-deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
from preprocess import extract_windows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main():
    BIG.mkdir(parents=True,exist_ok=True);BUNDLE.mkdir(exist_ok=True)
    rows=list(csv.DictReader((AUDIT/'usable-videos.csv').open(encoding='utf-8-sig')))
    assert len(rows)==881 and len({r['사람ID'] for r in rows})==64
    source=SOURCE/'cnn-multiwindow695/bundle/data.npz'
    with np.load(source) as z:d={k:z[k].copy() for k in z.files}
    assert len(d['y'])==1061 and len(set(d['person_id']))==692
    people=np.array(sorted({int(r['사람ID']) for r in rows}))
    perm=np.random.default_rng(20261104).permutation(people)
    mapping={int(p):f for f,g in enumerate(np.array_split(perm,5)) for p in g}
    xs=[];owners=[];starts=[];details=[]
    for i,r in enumerate(rows):
        p=Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104/poses')/r['좌표압축파일']
        frames=json.load(gzip.open(p,'rt'));raw=np.full((len(frames),75),np.nan)
        for k,frame in enumerate(frames):
            assert frame['frame']==k
            pp=frame['data']['people']
            if len(pp)==1:raw[k]=pp[0]['pose_keypoints_2d']
        finite=raw[:,2::3][np.isfinite(raw[:,2::3])]
        assert ((finite>=0)&(finite<=1)).all()
        x=extract_windows(raw)
        assert x['window_start'].tolist()==json.loads(r['유효구간시작_0부터'])
        assert len(x['x'])==int(r['유효124프레임구간수'])
        xs.append(x['x']);owners.extend([1061+i]*len(x['x']));starts.extend(x['window_start'])
        details.append({'source':'LJMU','source_id':r['영상ID'],'person_id':'LJMU:'+r['사람ID'],
                        'fold':mapping[int(r['사람ID'])],'age':int(r['나이']),
                        'label_kind':'assumed_100','left_gdi':100,'right_gdi':100,
                        'pose_sha256':sha(p),'source_video_sha256':r['SHA256']})
    nx=np.concatenate(xs);assert nx.shape==(3305,2,124,12)
    # Reconstructed source features must agree with the previously prepared data.
    old=ROOT/'gdi-research-harness/studies/20261031-ljmu-normal-cnn/input/normal_cnn_v2/prepared_normal/prepared.npz'
    with np.load(old) as z:
        lookup={str(v).removesuffix('_json'):i for i,v in enumerate(z['record_id'])}
        max_error=0.
        for i,r in enumerate(rows):
            stem=r['영상ID'].replace('_90_ALT','_90') if r['영상ID']=='P8_E2T6_90_ALT' else r['영상ID']
            # The archived record ID adds a dataset prefix; match the original suffix.
            candidates=[k for k in lookup if k.endswith(stem)]
            assert len(candidates)==1,(stem,candidates)
            ix=z['window_record']==lookup[candidates[0]]
            assert np.array_equal(z['window_start'][ix],json.loads(r['유효구간시작_0부터']))
            err=float(np.max(np.abs(z['x'][ix]-xs[i])))
            max_error=max(max_error,err)
        assert max_error<1e-5,max_error
    original_ids=d['person_id'].copy()
    out={
        'x':np.concatenate([d['x'],nx]),'window_record':np.concatenate([d['window_record'],owners]),
        'window_start':np.concatenate([d['window_start'],starts]),
        'canonical_fallback':np.concatenate([d['canonical_fallback'],np.zeros(len(nx),bool)]),
        'y':np.concatenate([d['y'],np.full((881,2),100.)]),
        'person_id':np.array(['Stanford:'+str(v) for v in original_ids]+[r['person_id'] for r in details]),
        'source_person_id':np.concatenate([original_ids,np.array([r['사람ID'] for r in rows])]),
        'video_id':np.array(['Stanford:'+str(v) for v in d['video_id']]+['LJMU:'+r['source_id'] for r in details]),
        'fold':np.concatenate([d['fold'],np.array([r['fold'] for r in details])]),
        'source':np.array(['Stanford']*1061+['LJMU']*881),
        'label_kind':np.array(['measured_gdi']*1061+['assumed_100']*881),
    }
    assert out['x'].shape==(14566,2,124,12) and out['y'].shape==(1942,2)
    splits=[]
    for f in range(5):
        test=out['fold']==f
        for j in range(5):
            if j==f:continue
            valid=out['fold']==j;train=~(test|valid)
            a,b,c=[set(out['person_id'][m]) for m in (train,valid,test)]
            assert not(a&b or a&c or b&c)
        splits.append({'fold':f,**{s:{'people':len(set(out['person_id'][test&(out['source']==s)])),
                                     'records':int((test&(out['source']==s)).sum())} for s in ['Stanford','LJMU']}})
    np.savez_compressed(BIG/'data.npz',**out)
    save(HERE/'ljmu-membership.json',details)
    families={'CNN':SOURCE/'cnn-multiwindow695/bundle','STT':SOURCE/'stt692/multiwindow/bundle','bilateral':SOURCE/'stt692/bilateral/bundle'}
    origins={}
    for family,src in families.items():
        dst=BUNDLE/family;dst.mkdir(exist_ok=True)
        for name in ['run.py','windows.py','protocol.json']+(['model.py'] if family=='CNN' else ['bilateral.py'] if family=='bilateral' else []):
            shutil.copyfile(src/name,dst/name);origins[str((dst/name).relative_to(BUNDLE))]=sha(src/name)
        if family!='CNN':
            for name in ['vendor','model']:shutil.copytree(src/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if (src/'sources').exists():shutil.copytree(src/'sources',dst/'sources',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        save(dst/'manifest.json',{n:sha(dst/n) for n in ['run.py','windows.py','protocol.json']})
    frozen=read(ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json')['members']
    members=[]
    for m in frozen:
        family=m['family'];tag=m.get('tag','refit');fold=m['fold']
        src=Path(m['checkpoint'].replace('C:\\Users\\jun\\Desktop\\D_P','D:\\Projects\\D_P'))
        assert sha(src)==m['checkpoint_sha256']
        target=BIG/'baseline'/family/f'fold{fold}_{tag}.pt';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        members.append(dict(m,checkpoint=str(target.relative_to(BIG)).replace('\\','/'),tag=tag))
    assert len(members)==54
    save(BUNDLE/'baseline-models.json',members)
    weights=read(SOURCE/'mean-difference692/analysis/selection.json')['folds']
    save(BUNDLE/'ensemble.json',[{k:r[k] for k in ['outer','cnn_weight','fraction']} for r in weights])
    protocol={
        'authorization':'User requested mixed Stanford + audited LJMU881 assumed100 training and person-disjoint before/after MAE comparison.',
        'arms':['Stanford_only_frozen','Stanford_plus_LJMU100'],
        'families':['CNN','STT','bilateral'],
        'reports':['CNN_refit','STT_refit','bilateral_refit','CNN_bag','STT_bag','fusion'],
        'data_counts':{'Stanford_people':692,'Stanford_records':1061,'LJMU_people':64,'LJMU_records':881,'windows':14566},
        'splits':splits,'LJMU_split_seed':20261104,
        'training':'Original architectures/optimizers/seeds and 5 outer x4 inner epoch selection. Add only LJMU people outside outer/inner held-out folds; fit 75 models. No final223 access.',
        'loss_weighting':'Original equal-record objective, inverse within-record window count. No dataset rebalancing. LJMU is 881/1942 records; not 64/756 of loss.',
        'epoch_selection':'Person-equal calibrated MAE pooled across internal validation people from both sources. Separate source metrics reported only for outer evaluation.',
        'calibration':'Per-model shared slope and intercept fitted on that model training records only, after averaging windows.',
        'ensemble':'Keep existing fold CNN weight and bilateral difference fraction fixed in both arms. Average eligible refit/inner CNN/STT; exclude a bag member iff training raw prediction SD<0.01 on either side, original rule.',
        'baseline':'Replay existing frozen54 checkpoint predictor per outer fold; each LJMU person evaluated by their assigned fold model only, not all-five-fold mean.',
        'primary_metric':'Within person mean absolute error across records and two sides, then equal mean across people, separately by source. LJMU MAE is deviation from assumed100, not measured GDI accuracy.',
        'comparison':'After minus before on exactly same held-out people; paired person bootstrap10000 seed20261104,95% interval. Stanford person mean trueGDI bands below60,60to75,75to90,90plus.',
        'validation':'Bundle hashes; all20 split boundaries; train-only scaling/calibration; selected epochs; original baseline replay; saved checkpoint replay tolerance0.001; independent metric reconstruction.',
        'training_initialization_seed':20261028,'baseline_source':str(SOURCE),'data_sha256':sha(BIG/'data.npz'),
    }
    save(BUNDLE/'study-protocol.json',protocol)
    save(HERE/'preparation.json',{'passed':True,'source_data_sha256':sha(source),'selection_sha256':sha(AUDIT/'usable-videos.csv'),
                               'data_sha256':sha(BIG/'data.npz'),'feature_replay_max_error':max_error,'splits':splits,'original_code_hashes':origins,
                               'large_storage':str(BIG),'estimated_storage_gb':2})
    print(json.dumps({'prepared':True,'shape':list(out['x'].shape),'splits':splits,'replay_error':max_error}),flush=True)

if __name__=='__main__':main()
