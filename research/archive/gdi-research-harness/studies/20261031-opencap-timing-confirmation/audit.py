from pathlib import Path
import sys,json,io,importlib.util,hashlib,collections,ast,urllib.request
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261031-opencap-preparation-offset'
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=mod('preparation_prior',PRIOR/'study.py')
np,pd,read,sha,TURN=old.np,old.pd,old.read,old.sha,old.TURN
def save(n,v):(HERE/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def trc(p):
    lines=p.read_text().splitlines();meta=lines[2].split();fps=float(meta[0]);assert meta[4]=='m'
    markers=[x for x in lines[3].split('\t')[2:] if x];data=np.loadtxt(io.StringIO('\n'.join(lines[6:])));xyz=data[:,2:].reshape(len(data),len(markers),3)
    assert len(data)==int(meta[2]) and len(markers)==int(meta[3]) and np.isfinite(xyz).all()
    return fps,data[:,1],np.array(markers),xyz
def author_bounds(fps,t,names,xyz):
    h=(xyz[:,list(names).index('RHJC_study'),1]+xyz[:,list(names).index('LHJC_study'),1])/2;h=(h-h.min())/(h.max()-h.min())
    if not any(h[:int(fps)]<.3) or not any(h[-int(fps):]<.3):return {'identifiable':False}
    start=int(np.argmax(h>.3));end=len(h)-int(np.argmax(h[::-1]>.7));end+=int(np.argmax(h[end:]<.3))
    # Execute only the public pure numerical function via AST, no import-time workflows.
    tree=ast.parse((HERE/'author-tug.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tug_trc_feats')
    import scipy.signal as ss
    scope={'np':np,'ss':ss,'norm':np.linalg.norm};exec(compile(ast.Module(body=[f],type_ignores=[]),'author-tug-function','exec'),scope)
    output=scope['tug_trc_feats'](xyz.copy(),names,fps);assert abs(output['tug_cone_time']-(end-start)/fps)<1e-10
    return {'identifiable':True,'start':float(t[start]),'end':float(t[end]),'duration':(end-start)/fps,'not_stopwatch_annotation':True}
def main():
    index=json.loads((HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())
    counts=collections.Counter(Path(x['name']).suffix.lower() or '(directory)' for x in index)
    media=[x['name'] for x in index if Path(x['name']).suffix.lower() in ['.mp4','.mov','.avi','.wav','.m4a','.mp3']]
    events=[x['name'] for x in index if any(s in Path(x['name']).name.lower() for s in ['annotation','events.tsv','stopwatch','timestamp','protocol','readme'])]
    clinical=HERE.parent/'20261030-opencap415-tug-timeseries/raw/participant_info.csv';df=pd.read_csv(clinical)
    rows=json.loads((PRIOR/'include283/manifest.json').read_text());trans={r['id']:r for r in json.loads((HERE.parent/'20261031-opencap-chair/transition-audit.json').read_text())};cases=[];sources=[]
    for m in rows:
        if m['id'] not in ['sub-283','sub-288']:continue
        target=df[(df.subid==m['id'])&(df.visit==m['visit'])];assert len(target)==1 and float(target.iloc[0].tft_tug)==m['target']
        raw=read(TURN/'raw'/m['turn_file']);assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256'];parent=Path(m['turn_file']).parent.parent.as_posix()
        yaml=next((HERE/'raw'/parent).glob('*.yaml'));text=yaml.read_text();keys=[line.split(':')[0] for line in text.splitlines() if line and not line[0].isspace()]
        for p in (HERE/'raw'/parent/'Markers').glob('*.trc'):
            fps,t,names,xyz=trc(p);bounds=author_bounds(fps,t,names,xyz)
            sources.append({'id':m['id'],'file':p.relative_to(HERE).as_posix(),'fps':fps,'time_start':float(t[0]),'time_end':float(t[-1]),'author_height_proxy':bounds,'sha256':sha(p)})
        cone=next(r for r in sources if r['id']==m['id'] and 'tug_cone' in r['file']);end=trans[m['id']]['sit']['window'][1];phase=m['phase'];start=phase['core'][0];prepstart=phase['extended'][0]
        cases.append({'id':m['id'],'visit':m['visit'],'target':m['target'],'selected_file':m['turn_file'],'raw_range':[float(raw['time'][0]),float(raw['time'][-1])],
          'metadata_keys':keys,'metadata_sha256':sha(yaml),'metadata_contains_timer_annotation':False,
          'candidate_start':prepstart,'core_rise_start':start,'rise_end':phase['core'][1],'sit_start':trans[m['id']]['sit']['window'][0],'sit_end':end,
          'candidate_to_sit_seconds':end-prepstart,'core_to_sit_seconds':end-start,'clinical_minus_candidate_span':m['target']-(end-prepstart),'clinical_minus_core_span':m['target']-(end-start),
          'label_derived_start_if_same_trial_and_sit_end_correct':end-m['target'],
          'IMPORTANT':'label-derived start is diagnostic only, circular if used to confirm onset or construct model input; not a measurement',
          'author_marker_proxy':cone['author_height_proxy'],'actual_timer_start':None,'actual_timer_end':None,'timed_trial_confirmed':False,'preparation_in_timer_confirmed':None})
    save('archive-content-audit.json',{'entries':len(index),'file_extensions':dict(counts),'media_files':media,'timing_annotation_name_candidates':events,'clinical_columns':list(df.columns),'clinical_md5':hashlib.md5(clinical.read_bytes()).hexdigest(),'clinical_sha256':sha(clinical),'archive_index_sha256':sha(HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json')})
    save('marker-boundaries.json',sources);save('case-timing-audit.json',cases)
    annotations=[{'id':m['id'],'visit':m['visit'],'file':m['turn_file'],'file_sha256':m['turn_sha256'],'clinical_timing_status':'unverified','timer_start':None,'timer_end':None,'confirmed_preparation_intervals':None,'evidence':None} for m in rows]
    save('timing-annotations.json',annotations)
    save('confirmed-experiment-status.json',{'status':'not_executable_without_external_timing_evidence','people_with_verified_timer_and_trial':0,'requested_people':len(rows),'not_zero_true_preparation':True,'no_target_derived_onsets_allowed':True})
    print(json.dumps({'archive_extensions':dict(counts),'media_count':len(media),'event_name_matches':events,'cases':cases,'marker_sources':sources},indent=2),flush=True)
if __name__=='__main__':main()
