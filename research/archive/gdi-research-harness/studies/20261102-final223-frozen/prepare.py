"""Apply fixed development preprocessing to the reserved membership, without labels."""
from common import *
import struct,zlib,warnings,time
from collections import Counter
source=module('dev_multi_prepare',STUDY/'cnn-multiwindow695/prepare_inputs.py')
ns=source.source_functions()
arc=module('bounded_archive',ROOT/'tools/stanford-sample-check/inspect_archive.py')
LEFT=[28,29,26,27,24,25,38,39,50,52,54,56]
RIGHT=[22,23,20,21,18,19,44,45,51,53,55,57]

def convert(raw,ids):
    observed=np.flatnonzero(np.isfinite(raw).any(axis=1))
    if len(observed):raw=raw[:observed[-1]+1]
    reason=None;processed=None;missing=None;canonical=None;starts=[]
    if not len(raw) or ids.min()<1 or ids.max()>len(raw):reason='matched_frames_outside_official_converter_extent'
    else:
        missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[ids-1]),[14,11,13,10,12,9,19,22]))
        if missing>.25:reason='matched_window_over_25pct_missing_required_coordinates'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
            canonical=np.stack([processed[ids-1][:,LEFT],processed[ids-1][:,RIGHT]])
            if not np.isfinite(canonical).all():reason='nonfinite_required_feature_after_official_processing'
    if processed is not None:
        for k in range(0,500-124,31):
            chunk=raw[k:k+124]
            if len(chunk)!=124:continue
            miss=ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](chunk),[14,11,13,10,12,9,19,22])
            xx=np.stack([processed[k:k+124,LEFT],processed[k:k+124,RIGHT]])
            if miss<=.25 and np.isfinite(xx).all():starts.append(k)
    fallback=reason is None and not starts
    if fallback:starts=[int(ids[0])-1]
    xx=None if reason else np.stack([np.stack([processed[k:k+124,LEFT],processed[k:k+124,RIGHT]]) for k in starts]).astype(np.float32)
    return xx,dict(eligible=reason is None,exclusion_reason=reason,missing_fraction=missing,raw_frames=len(raw),starts=starts,canonical_fallback=fallback),canonical

def scan(rows,callback):
    byfolder={'openpose/'+r['video_id']+'-processed':r for r in rows}
    found=set();start=time.monotonic()
    with arc.ARCHIVE.open('rb',buffering=4*1024*1024) as directory,arc.ARCHIVE.open('rb') as data:
        count,_,offset=arc.directory_info(directory);directory.seek(offset)
        current=None;members={}
        def process():
            if current not in byfolder:return
            row=byfolder[current];last=0
            while last+1 in members and last<1499:last+=1
            raw=np.full((last,75),np.nan,dtype=np.float64);digest=hashlib.sha256();ambiguous=0
            for frame in range(1,last+1):
                m=members[frame];data.seek(m['offset']);header=data.read(30);assert header[:4]==b'PK\x03\x04'
                nl,el=struct.unpack_from('<HH',header,26);assert data.read(nl).decode()==m['name'];data.seek(el,1)
                packed=data.read(m['compressed']);content=zlib.decompress(packed,-15) if m['method']==8 else packed
                assert len(content)==m['size'] and zlib.crc32(content)&0xffffffff==m['crc'];digest.update(content)
                candidates=[]
                for person in json.loads(content)['people']:
                    a=np.array(person['pose_keypoints_2d'],dtype=np.float64)
                    if a.shape==(75,) and np.isfinite(a).all() and a[::3].max()<320:candidates.append(a)
                if len(candidates)==1:raw[frame-1]=candidates[0]
                elif len(candidates)>1:ambiguous+=1
            callback(row,raw,digest.hexdigest(),ambiguous);found.add(row['video_id'])
            if len(found)%25==0:print(json.dumps({'prepared':len(found),'target':len(rows),'seconds':round(time.monotonic()-start,1)}),flush=True)
        for _ in range(count):
            header=directory.read(46);assert header[:4]==b'PK\x01\x02';f=struct.unpack('<4s6H3L5H2L',header)
            name=directory.read(f[10]).decode();directory.seek(f[11]+f[12],1);group='/'.join(name.split('/')[:2])
            if group!=current:
                process();current=group;members={}
                if len(found)==len(rows):break
            if group in byfolder and name.endswith('_keypoints.json'):
                frame=int(name.rsplit('_',2)[-2])
                if frame<=1499:
                    assert not f[3]&1 and f[4] in (0,8)
                    members[frame]=dict(name=name,offset=f[16],size=f[9],compressed=f[8],crc=f[7],method=f[4])
        if len(found)<len(rows):process()
    assert found==set(r['video_id'] for r in rows)

def main():
    locked();assert not (HERE/'predictions-locked.json').exists()
    # First replay three development records using the same raw archive and exact features.
    devrows=read(STUDY/'cnn-multiwindow695/input_sources.json')['records'][:3];replays=[]
    def replay(row,raw,digest,ambiguous):
        ids=np.array(row['frames']);xx,q,canonical=convert(raw,ids)
        with np.load(STUDY/('cnn-multiwindow695/records/%04d.npz'%row['index'])) as z:
            assert np.array_equal(xx,z['multi_x']) and np.array_equal(np.asarray(q['starts']),z['window_start'])
            assert np.array_equal(canonical.astype(np.float32),z['x'])
        replays.append(row['index'])
    scan(devrows,replay)
    old=H/'studies/20261027-window-ablation';receipt=read(old/'study_receipt.json')
    sealed={r['path']:r['sha256'] for r in receipt['artifacts']}
    for name in ['new_person_split.json','new_record_split.json','new_cohort_summary.json']:
        p=old/name;assert sha(p)==sealed[p.relative_to(H).as_posix()]
    split=read(old/'new_person_split.json');people={r['patient_id'] for r in split['people'] if r['reserved_role']=='final_evaluation_reserved'}
    rows=[r for r in read(old/'new_record_split.json')['records'] if r['reserved_role']=='final_evaluation_reserved' and r['input124_status']=='prepared']
    linkage=read(H/'studies/20261027-gdi-linkage/final_linkage_presence_only.json')['records']
    lookup={r['video_id']:r for r in linkage}
    assert len(rows)==378 and len(people)==223 and {r['patient_id'] for r in rows}==people
    with np.load(STUDY/'cnn-multiwindow695/bundle/data.npz') as z:dev_people=set(z['person_id'].astype(str))
    assert len(dev_people)==692 and not people&dev_people
    model_audit=[]
    for m in read(HERE/'models.json')['members']:
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True);identity=ck['identity']
        tr=set(map(str,identity['train_people']));va=set(map(str,identity['validation_people']))
        assert not people&(tr|va) and tr<=dev_people and va<=dev_people
        model_audit.append(dict(checkpoint_sha256=m['checkpoint_sha256'],train_people=len(tr),validation_people=len(va),reserved_overlap=0))
    for i,r in enumerate(rows):
        assert r['patient_id']==lookup[r['video_id']]['patient_id']
        p=H/r['artifact'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False) as z:ids=z['frame_ids124'].copy()
        assert ids.shape==(124,) and (np.diff(ids)==1).all()
        r.update(index=i,frames=ids.tolist(),exam_id=lookup[r['video_id']]['exam_id'])
    stat=dict(size=arc.ARCHIVE.stat().st_size,mtime_ns=arc.ARCHIVE.stat().st_mtime_ns)
    assert stat==read(H/'jobs/native124-cache-v1/config.json')['archive_stat']
    save(HERE/'input-membership.json',dict(records=rows,people=sorted(people),source_hashes={str(old/name):sha(old/name) for name in ['new_person_split.json','new_record_split.json','new_cohort_summary.json']},archive_stat=stat))
    proofs=[]
    def prepare(row,raw,digest,ambiguous):
        p=BIG/('records/%04d.npz'%row['index']);meta=p.with_suffix('.json')
        if meta.exists():
            proof=read(meta);assert proof['raw_content_sha256']==digest and proof['protocol_sha256']==sha(HERE/'protocol.json')
            if proof['eligible']:assert sha(p)==proof['features_sha256']
        else:
            xx,q,_=convert(raw,np.array(row['frames']))
            proof=dict(**q,index=row['index'],video_id=row['video_id'],person_id=row['patient_id'],ambiguous_frames=ambiguous,raw_content_sha256=digest,protocol_sha256=sha(HERE/'protocol.json'))
            if q['eligible']:
                np.savez_compressed(p,x=xx,window_start=np.array(q['starts'],dtype=np.int64));proof['features_sha256']=sha(p)
            save(meta,proof)
        proofs.append(proof)
    scan(rows,prepare);proofs.sort(key=lambda p:p['index']);parts=[];owners=[];starts=[];eligible=[]
    for p in proofs:
        if not p['eligible']:continue
        r=rows[p['index']];j=len(eligible);eligible.append(r)
        with np.load(BIG/('records/%04d.npz'%p['index'])) as z:
            parts.append(z['x']);starts.extend(z['window_start'].tolist());owners.extend([j]*len(z['x']))
    np.savez_compressed(BIG/'features.npz',x=np.concatenate(parts),window_record=np.array(owners),window_start=np.array(starts),person_id=np.array([r['patient_id'] for r in eligible]),video_id=np.array([r['video_id'] for r in eligible]))
    assert stat==dict(size=arc.ARCHIVE.stat().st_size,mtime_ns=arc.ARCHIVE.stat().st_mtime_ns)
    audit=dict(reserved_people=223,reserved_records=378,eligible_people=len({r['patient_id'] for r in eligible}),eligible_records=len(eligible),windows=len(owners),fallback_records=sum(p['canonical_fallback'] for p in proofs if p['eligible']),exclusions=dict(Counter(p['exclusion_reason'] for p in proofs if not p['eligible'])),quality=proofs,development_replay_records=replays,development_feature_replay_exact=True,model_identity_audit=model_audit,protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),numeric_reserved_labels_opened=False)
    save(HERE/'input-audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['quality','model_identity_audit']}),flush=True)
if __name__=='__main__':main()
