"""Freeze equal people/record/window counts and equal external person mass."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
OLD=HERE.parent/'20261105-child100-train-only'
SOURCE=ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
POSE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
sys.path[:0]=[str(ROOT/'tmp/ljmu-analysis-python-deps'),str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
from preprocess import extract_windows
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def rank(value):return hashlib.sha256(('20261105:'+value).encode()).hexdigest()
review=read(HERE/'live-review.json');candidates=read(HERE/'live-candidates.json');children=[];xs=[];rejected=[]
for row in candidates:
    stem=row['record']
    if stem not in review['accepted']:continue
    decision=review['accepted'][stem];c=dict(row['candidates'][decision['candidate']]);picks=c['person_indices'].copy()
    if 'start_override' in decision:
        mapping=dict(zip(range(c['start'],c['end']),picks));c['start']=decision['start_override'];c['end']=c['start']+124
        picks=[mapping.get(f,-1) for f in range(c['start'],c['end'])]
    for f in decision.get('mask_frames',[]):
        if c['start']<=f<c['end']:picks[f-c['start']]=-1
    raw=np.full((124,75),np.nan);digests=[]
    for k,sel in enumerate(picks):
        path=POSE/stem/('%s_%012d_keypoints.json'%(stem,c['start']+k));digests.append(sha(path))
        if sel>=0:raw[k]=read(path)['people'][sel]['pose_keypoints_2d']
    try:ex=extract_windows(raw);assert ex['window_start'].tolist()==[0]
    except ValueError as e:rejected.append(dict(record=stem,reason=str(e)));continue
    # Re-evaluate movement after any manually excluded wrong-target frames.
    a=raw.reshape(124,25,3);valid=np.isfinite(a[:,8,0])&(a[:,8,2]>.1);obs=np.flatnonzero(valid)
    hip=np.interp(np.arange(124),obs,a[obs,8,0]);pad=np.pad(hip,(4,4),mode='edge');hip=np.median(np.stack([pad[k:k+124] for k in range(9)]),axis=0)
    thigh=np.linalg.norm(a[:,[9,12],:2]-a[:,[10,13],:2],axis=2);scale=np.nanmedian(thigh[(a[:,[9,12],2]>.1)&(a[:,[10,13],2]>.1)])
    moving=(np.abs(np.gradient(hip))>max(1,.015*scale))&valid
    if moving.mean()<.8:rejected.append(dict(record=stem,reason='Repaired target movement fraction below unchanged80%',fraction=float(moving.mean())));continue
    xs.append(ex['x']);children.append(dict(record=stem,person='LIVE-TD:'+row['person'],age=row['age'],start=c['start'],end=c['end'],native_fps=row['native_fps'],
        missing_fraction=float(ex['window_missing_fraction'][0]),moving_fraction=float(moving.mean()),person_indices=picks,
        video_sha256=sha(SOURCE/'td_sagittal'/(stem+'.MOV')),pose_sequence_sha256=hashlib.sha256(''.join(digests).encode()).hexdigest(),review=decision))
assert children
with np.load(Path('D:/gait-research-data/D_P/child100-train-only-20261105/data.npz')) as z:
    # Reuse only original Stanford prefix, never earlier child features.
    d={k:z[k][:11261 if k in ['x','window_record','window_start','canonical_fallback'] else 1061].copy() for k in z.files}
assert np.all(d['source']=='Stanford') and len(set(d['person_id']))==692
adult_path=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104/data.npz')
assert sha(adult_path)==read(HERE.parent/'20261104-ljmu100-mixed/bundle/study-protocol.json')['data_sha256']
with np.load(adult_path) as z:ad={k:z[k] for k in z.files}
adult_meta=read(HERE.parent/'20261104-ljmu100-mixed/ljmu-membership.json');adult_meta={r['source_id']:r for r in adult_meta}
child_people=sorted({r['person'] for r in children});adult_people=sorted(set(ad['person_id'][ad['source']=='LJMU']),key=rank)[:len(child_people)]
adults=[];ax=[]
for cp,ap in zip(child_people,adult_people):
    records=sorted(np.flatnonzero(ad['person_id']==ap),key=lambda i:rank(str(ad['video_id'][i])))
    count=sum(r['person']==cp for r in children);assert len(records)>=count
    for i in records[:count]:
        wi=np.flatnonzero(ad['window_record']==i);w=int(wi[len(wi)//2])
        key=str(ad['video_id'][i]).removeprefix('LJMU:');meta=adult_meta[key]
        adults.append(dict(record=key,person=str(ap),age=meta['age'],matched_child=cp,source_row=int(i),source_window=w,start=int(ad['window_start'][w]),selection='Hash-ranked people and recordings; middle eligible window. No GDI predictions consulted.',**{k:meta[k] for k in ['pose_sha256','source_video_sha256']}));ax.append(ad['x'][w:w+1])
assert len(adults)==len(children)
counts={};hashes={};splits=[]
for arm,rows,extra_x in [('child',children,np.concatenate(xs)),('adult',adults,np.concatenate(ax))]:
    target=BIG/arm/'data.npz'
    if target.exists():
        assert '--rebuild-unsealed' in sys.argv and not (HERE/'bundle/manifest.json').exists() and not (BIG/arm/'runs').exists(),'Preserve frozen datasets'
    n=len(rows);extraids=[r['person'] for r in rows];weights=np.array([1/extraids.count(p) for p in extraids])
    out={k:v.copy() for k,v in d.items()}
    for k,v in dict(x=extra_x,window_record=np.arange(1061,1061+n),window_start=np.zeros(n,dtype=np.int64),canonical_fallback=np.zeros(n,bool),y=np.full((n,2),100.),person_id=np.array(extraids),source_person_id=np.array([p.split(':')[1] for p in extraids]),video_id=np.array([('LJMU:' if arm=='adult' else 'LIVE-TD:')+r['record'] for r in rows]),fold=np.full(n,-1,dtype=np.int64),source=np.array(['LJMU' if arm=='adult' else 'LIVE-TD']*n),label_kind=np.array(['assumed_100']*n),age=np.array([r['age'] for r in rows])).items():out[k]=np.concatenate([out[k],v])
    out['record_weight']=np.concatenate([np.ones(1061),weights]);external=out['fold']==-1
    for f in range(5):
        for j in [None]+[j for j in range(5) if j!=f]:
            tr=(out['fold']!=f) if j is None else (out['fold']!=f)&(out['fold']!=j);te=out['fold']==f;va=np.zeros(len(tr),bool) if j is None else out['fold']==j
            assert tr[external].all() and not (te|va)[external].any()
            assert not set(out['person_id'][tr])&(set(out['person_id'][te])|set(out['person_id'][va]))
            for p in np.unique(out['person_id'][external]):assert abs(out['record_weight'][out['person_id']==p].sum()-1)<1e-12
            splits.append(dict(arm=arm,outer=f,inner=j,Stanford_training_records=int((tr&~external).sum()),external_people=len(child_people),external_weight_share=float(out['record_weight'][external].sum()/out['record_weight'][tr].sum())))
    for k in d:assert np.array_equal(out[k][:len(d[k])],d[k]),k
    target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**out);hashes[arm]=sha(target)
    counts[arm]=dict(records=len(out['y']),extra_records=n,extra_windows=n,extra_people=len(child_people),total_windows=len(out['x']))
save(HERE/'membership.json',dict(child=children,adult=adults,preprocessing_rejected=rejected,review_sha256=sha(HERE/'live-review.json'),adult_input_sha256=sha(adult_path)))
protocol=dict(status='frozen before training',matched_people=len(child_people),counts=counts,data_sha256=hashes,membership_sha256=sha(HERE/'membership.json'),
    arms=['Stanford_only_frozen','adult','child'],reports=['CNN_refit','RF','STT_refit','CNN_bag','STT_bag','bilateral_refit','fusion'],
    training='Stanford original equal-record objective retained. Each external person has total weight1 divided across their recordings; one124-frame window per external recording. Exact matched people, records, windows and per-person mass across arms. Native frames; no stretching, repetition or joining.',
    calibration='Neural shared OLS and target scale use Stanford training rows only at EVERY epoch and refit. RF retains original uncalibrated output.',
    selection='All external people train-only fold=-1; epoch and RF candidate selection use Stanford inner overall equal-person MAE only. Same original5 outer folds, seeds, max epochs, architecture. Bag-collapse SD uses Stanford training rows only. Frozen ensemble weights.',
    metrics='Stanford measured-GDI only. Overall, record-side>=90,>=100,<70; person-equal MAE, prediction-truth bias, max(truth-prediction,0) shortfall, paired10000 bootstrap seed20261105. Final223 not accessed.',
    limits='Single deterministic adult sample and one initialization schedule; domain/camera/disease differences remain. Matching volume does not isolate a purely biological age effect. All added100labels assumed.')
save(HERE/'bundle/study-protocol.json',protocol)
rf=read(OLD/'rf-protocol.json');rf.update(data_sha256=hashes,cohort=protocol['selection'],training=protocol['training']+' RF sample_weight duplicates each record weight for both sides; no OLS or scaling.',metric=protocol['metrics'],fits_per_arm=85);save(HERE/'rf-protocol.json',rf)
save(HERE/'preparation.json',dict(passed=True,counts=counts,all50_splits=splits,Stanford_arrays_identical=True,external_weights_match=True))
print(json.dumps(dict(counts=counts,rejected=rejected,adult_ages=[r['age'] for r in adults]),indent=2))
