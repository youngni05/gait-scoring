"""Build a new training-only augmentation without modifying the Stanford cohort."""
import csv, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/child100-train-only-20261105')
SOURCE = ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
POSE = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
OLD = ROOT/'studies/20261104-ljmu100-mixed'
sys.path[:0] = [str(ROOT/'tmp/ljmu-analysis-python-deps'), str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
from preprocess import extract_windows

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def main():
    BIG.mkdir(parents=True, exist_ok=True)
    assert not (BIG/'data.npz').exists(), 'Preserve the sealed dataset; prepare a new study revision instead.'
    candidates = read(HERE/'segment-candidates.json')
    source_info = {Path(r['file']).stem:r for r in read(SOURCE/'td_download.json')['videos']}
    # Manual video review precedes child100 fitting. This is not prediction-based selection.
    chosen = {
        'S04-240404-C0-GAIT-02-SAGITTAL', 'S09-240425-C0-GAIT-01-SAGITTAL',
        'S09-240425-C0-GAIT-02-SAGITTAL', 'S10-240425-C0-GAIT-02-SAGITTAL',
        'S11-240521-C0-GAIT-02-SAGITTAL', 'S15-240606-C0-GAIT-01-SAGITTAL',
        'S15-240606-C0-GAIT-02-SAGITTAL', 'S17-240618-C0-GAIT-02-SAGITTAL',
        'S8-240425-C0-GAIT-01-SAGITTAL', 'S8-240425-C0-GAIT-02-SAGITTAL',
    }
    accepted, rejected, xs = [], [], []
    for row in candidates:
        stem = row['record']
        if stem not in chosen:
            reason = ('jump before walking' if stem.startswith('S16-') else
                      'discontinuous passage in candidate window' if stem.startswith('S20-') else
                      'movement-screen coverage below 80%; not manually accepted')
            rejected.append(dict(record=stem, reason=reason)); continue
        start, end = row['best_start'], row['best_end_exclusive']
        folder = POSE/stem
        picks = {int(r['filename'].split('_')[-2]):int(r['person_index'])
                 for r in csv.DictReader((folder/'selection.csv').open())}
        raw = np.full((124,75),np.nan); evidence=[]
        for k, f in enumerate(range(start,end)):
            p = folder/('%s_%012d_keypoints.json'%(stem,f))
            evidence.append(sha(p))
            sel = picks[f]
            if sel >= 0: raw[k] = read(p)['people'][sel]['pose_keypoints_2d']
        try:
            extracted = extract_windows(raw)
            assert extracted['window_start'].tolist() == [0]
        except ValueError as exc:
            rejected.append(dict(record=stem, reason='frozen preprocessing rejected: '+str(exc))); continue
        xs.append(extracted['x'])
        accepted.append(dict(row, review_status='accepted_visual_and_frozen_preprocessing',
            label_kind='assumed_100', left_gdi=100, right_gdi=100, fold=-1,
            required_coordinate_missing_fraction=float(extracted['window_missing_fraction'][0]),
            selection_sha256=sha(folder/'selection.csv'),
            pose_sequence_sha256=hashlib.sha256(''.join(evidence).encode()).hexdigest(),
            video_sha256=sha(SOURCE/'td_sagittal'/(stem+'.MOV')),
            review_note='Straight sagittal passage. Short entry/exit occlusion allowed only by unchanged 25% missing-coordinate rule. No temporal stretching, looping, or joining passes.'))
    save(HERE/'child-membership.json', {'accepted':accepted, 'rejected':rejected,
        'universe':'31 previously identity-reviewed clips; no exhaustive recovery of remaining source clips',
        'selection':'One maximum-motion candidate per clip; movement proxy >=80% plus visual review plus frozen preprocessing. Motion proxy is not a validated gait classifier.'})
    assert accepted, 'No eligible walking clips'
    src = ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
    with np.load(src) as z: d = {k:z[k].copy() for k in z.files}
    assert d['x'].shape == (11261,2,124,12) and len(d['y']) == 1061
    n = len(accepted); child_x = np.concatenate(xs)
    agefile = ROOT/'gdi-research-harness/studies/20261031-ljmu-normal-cnn/diagnostics/development_age_metadata.npz'
    with np.load(agefile) as z:
        assert np.array_equal(d['person_id'].astype(str),z['person_id'].astype(str))
        age = z['age'].copy()
    out = dict(
        x=np.concatenate([d['x'],child_x]),
        window_record=np.concatenate([d['window_record'],np.arange(1061,1061+n)]),
        window_start=np.concatenate([d['window_start'],np.zeros(n,dtype=np.int64)]),
        canonical_fallback=np.concatenate([d['canonical_fallback'],np.zeros(n,dtype=bool)]),
        y=np.concatenate([d['y'],np.full((n,2),100.)]),
        person_id=np.array(['Stanford:'+str(v) for v in d['person_id']]+['LIVE-TD:'+r['person'] for r in accepted]),
        source_person_id=np.concatenate([d['person_id'].astype(str),np.array([r['person'] for r in accepted])]),
        video_id=np.array(['Stanford:'+str(v) for v in d['video_id']]+['LIVE-TD:'+r['record'] for r in accepted]),
        fold=np.concatenate([d['fold'],np.full(n,-1,dtype=np.int64)]),
        source=np.array(['Stanford']*1061+['LIVE-TD']*n),
        label_kind=np.array(['measured_gdi']*1061+['assumed_100']*n),
        age=np.concatenate([age,np.array([r['age'] for r in accepted])]))
    for f in range(5):
        for j in [None]+[j for j in range(5) if j != f]:
            te=out['fold']==f; va=np.zeros(len(te),bool) if j is None else out['fold']==j; tr=~(te|va)
            assert tr[1061:].all() and not (te[1061:].any() or va[1061:].any())
            a,b,c=[set(out['person_id'][m]) for m in (tr,va,te)]
            assert not (a&b or a&c or b&c)
            assert np.array_equal(te[:1061],d['fold']==f)
    for k in ('x','y','fold','window_record','window_start','canonical_fallback'):
        assert np.array_equal(out[k][:len(d[k])],d[k]), k
    np.savez_compressed(BIG/'data.npz',**out)
    counts=dict(Stanford_people=692,Stanford_records=1061,child_people=len({r['person'] for r in accepted}),
                child_records=n,windows=len(out['x']))
    protocol=dict(status='declared before child100 model fitting',authorization='User requested healthy-child assumed100 added to training only, testing original held-out high-GDI accuracy.',
        data_counts=counts,data_sha256=sha(BIG/'data.npz'),membership_sha256=sha(HERE/'child-membership.json'),
        arms=['Stanford_only_frozen','Stanford_plus_child100_training_only'],
        reports=['CNN_refit','RF','STT_refit','CNN_bag','STT_bag','bilateral_refit','fusion'],
        split='Same five Stanford outer folds and four inner folds. All accepted children in training of every fit (fold=-1); no child in validation or evaluation. No final223 access.',
        labels='Child 100 is an experimental assumption, not measured GDI.',
        primary_metric='Actual Stanford record-side GDI >=90 and >=100. Average eligible side-record absolute errors within each person, then equal-person mean. Also signed prediction minus truth and paired person bootstrap 10000 seed20261105.',
        guardrails='Report whole-cohort and true side GDI <70 MAE. Supplementary age5<=age<13 based on contemporaneous record age.',
        training='Original architectures, optimizers, seeds, maximum epochs, equal-record loss and training-only shared OLS calibration. No child duplication or extra weighting.',
        selection='Epoch/RF hyperparameter selection uses Stanford inner-validation overall person MAE only. No outer high-score outcome used.',
        ensemble='Keep original fold weights and bag-collapse rule fixed. No outer-data weight tuning.',
        limitation='Small younger 5-10-year-old subset and fixed camera; not a full age match to Stanford. Adult LJMU experiment uses different validation inclusion and sample volume, so cross-experiment differences are not an isolated age effect.',
        qa='Walking-centered124-frame crops retain original time and BODY25 feature extraction. Missing coordinates use unchanged <=25% threshold; some entry/exit occlusion remains.')
    bundle=HERE/'bundle';bundle.mkdir(exist_ok=True)
    for family in ['CNN','STT','bilateral']:
        shutil.copytree(OLD/'bundle'/family,bundle/family,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['ensemble.json','baseline-models.json']:shutil.copyfile(OLD/'bundle'/name,bundle/name)
    save(bundle/'study-protocol.json',protocol)
    rf=read(OLD/'rf-protocol.json');rf.update(data_sha256=protocol['data_sha256'],cohort=protocol['split'],
        training='Shared RF on equal-weight record-side rows. No feature scaling, clipping or OLS calibration. All child rows occur in training only.',
        metric=protocol['primary_metric'],selection=protocol['selection'],fits=85,
        baseline='Reuse already completed Stanford-only RF with identical folds and unchanged original features.')
    save(HERE/'rf-protocol.json',rf)
    save(HERE/'preparation.json',dict(passed=True,counts=counts,source_data_sha256=sha(src),
        data_sha256=protocol['data_sha256'],all25_splits_child_training_only=True,Stanford_arrays_identical=True,
        accepted=[r['record'] for r in accepted],rejected_preprocessing=[r for r in rejected if 'preprocessing' in r['reason']]))
    print(json.dumps(read(HERE/'preparation.json'),indent=2))

if __name__ == '__main__': main()
