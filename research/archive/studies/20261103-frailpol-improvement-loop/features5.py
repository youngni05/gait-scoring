"""Outcome-blind bilateral short-window measurement; preserve original anchor/cohort."""
from pathlib import Path
import sys,json,zipfile,warnings,argparse,time
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
import features2 as f2
from extract import heading,calculate,fsf,convert_to_fbf,BarthDtw,BarthOriginalTemplate,HerzerEventDetection,np
from signal_pilot import filemap,rawread,save
B=f2.B;SRC=f2.SRC
KIN=['speed','stride_time','swing_fraction','stride_length','max_sensor_lift','max_lateral_excursion','max_orientation_change','ic_angle','tc_angle','ankle_gyro_rms','ankle_acc_std']
def select(raw,original):
    side=next(iter(original['selected_events']));e=original['selected_events'][side]
    a=fsf(raw);bf=convert_to_fbf(a,left_like='left',right_like='right')
    with warnings.catch_warnings(record=True):
        dtw=BarthDtw(template=BarthOriginalTemplate(use_cols=['gyr_ml'])).segment(bf,sampling_rate_hz=100)
        ed=HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(bf,stride_list=dtw.stride_list_,sampling_rate_hz=100)
    events={s:d.reset_index().to_dict('records') for s,d in ed.min_vel_event_list_.items()}
    found=[q for q in events[side] if q['start']==e['start'] and q['end']==e['end']];assert len(found)==1
    assert all(found[0][k]==e[k] for k in ['start','end','tc','ic'])
    accepted=[dict(side=side,event=e)];last={side:e};lo=int(e['start']);hi=int(e['end']);h,horiz=heading(raw);stop='no_next_overlapping_cycle'
    candidates=sorted([(s,q) for s,ee in events.items() for q in ee if q['start']>=lo and q['end']>=hi and not(s==side and q['start']==lo and q['end']==hi)],key=lambda sq:(sq[1]['end'],sq[1]['start'],sq[0],sq[1]['s_id']))
    for s,q in candidates:
        if not(q['start']<q['tc']<q['ic']<q['end'] and .6<=(q['end']-q['start'])/100<=3):stop='invalid_next_events';break
        if q['start']>hi:stop='gap_after_selected_cycles';break
        if s in last and q['start']!=last[s]['end']:stop='noncontiguous_same_foot';break
        nxt=int(max(hi,q['end']))
        if nxt-lo>400 or nxt+9>len(h):stop='four_second_cap';break
        if np.ptp(h[lo:nxt])>20 or np.min(horiz[lo:nxt])<.5:stop='turn_or_orientation';break
        try:calculate(raw,{s:q})
        except Exception as exc:stop='trajectory_failed:'+str(exc);break
        accepted.append(dict(side=s,event=q));last[s]=q;hi=nxt
    return accepted,stop
def render(raw,accepted,original):
    per={};lo=min(int(q['event']['start']) for q in accepted);hi=max(int(q['event']['end']) for q in accepted)
    for item in accepted:
        v,_=calculate(raw,{item['side']:item['event']});per.setdefault(item['side'],[]).append(v)
    base={k:float(np.mean([np.mean([v[k] for v in vv]) for vv in per.values()])) for k in KIN}
    # Trunk movement summarizes exactly this selected core, not the complete TUG.
    sac=raw['sacrum back'].iloc[lo:hi]
    for kind in ['Acc','Gyr']:
        x=sac[[kind+'_'+a for a in ['X','Y','Z']]].to_numpy(float)
        if kind=='Gyr':x=np.rad2deg(x)
        for j,axis in enumerate(['X','Y','Z']):base[f'sacrum_{kind}_{axis}_std']=float(np.std(x[:,j]))
        base[f'sacrum_{kind}_norm_std']=float(np.std(np.linalg.norm(x,axis=1)))
    derived={}
    for k in KIN:
        means=[float(np.mean([v[k] for v in vv])) for vv in per.values()];derived['bilateral_asymmetry_'+k]=abs(means[0]-means[1])/(.5*(abs(means[0])+abs(means[1]))+1e-6) if len(means)==2 else None
        dof=sum(len(vv)-1 for vv in per.values());ss=sum(sum((v[k]-np.mean([q[k] for q in vv]))**2 for v in vv) for vv in per.values());den=float(np.mean([abs(v[k]) for vv in per.values() for v in vv]))+1e-6
        derived['within_foot_relative_std_'+k]=float(np.sqrt(ss/dof)/den) if dof>0 else None
    phase=[]
    for item in accepted:
        q=item['event'];others=[r['event']['start'] for r in accepted if r['side']!=item['side'] and q['start']<r['event']['start']<q['end']]
        if len(others)==1:phase.append((others[0]-q['start'])/(q['end']-q['start']))
    derived['alternating_min_velocity_phase_deviation']=float(np.mean(np.abs(np.array(phase)-.5))*2) if phase else None
    derived['alternating_min_velocity_phase_std']=float(np.std(phase,ddof=1)) if len(phase)>1 else None
    derived['observed_feet']=len(per);derived['short_window_cycle_count']=len(accepted)
    window=dict(core_start=lo,core_end=hi,support_start=lo-8,support_end_exclusive=hi+9,support_seconds=(hi-lo+17)/100)
    extra,bad=f2.features(raw,window)
    return dict(features={**base,**extra,**derived},window=window,audit=dict(cycles=len(accepted),foot_counts={k:len(v) for k,v in per.items()},phase_count=len(phase),invalid_optional_sensors=bad,selected_events=accepted))
def one(raw,original):
    try:accepted,stop=select(raw,original)
    except Exception as exc:accepted=[dict(side=s,event=e) for s,e in original['selected_events'].items()];stop='original_anchor_fallback:'+str(exc)
    r=render(raw,accepted,original);r['audit']['stop_reason']=stop;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args();h=H/'round05';b=B/'round05';h.mkdir(exist_ok=True);b.mkdir(exist_ok=True)
    proto=dict(code_sha256=f2.sha(Path(__file__)),library_sha256=f2.sha(H/'features2.py'),extractor_sha256=f2.sha(Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/extract.py')),source_sha256=f2.sha(SRC/'features.json'),cohort_sha256=f2.sha(SRC/'cohorts.json'),raw_sha256=f2.sha(SRC/'raw/IMU_signals.zip'),design='Same594 IDs/anchor. Greedy end-ordered overlapping or contiguous complete strides from bothfeet starting at/after original anchor start. Same-foot adjacency exact. Stop on first invalid/gap/turn/plausibility failure or4s core cap. Equal weight each observed foot for11kinematic means. Trunk descriptors recomputed in selected core. Same168sensor descriptors. Add11relative bilateral asymmetries,11pooled-within-foot relative standard deviations, min-velocity alternation phase deviation/std, observed-foot and short-window-cycle counts. Missing second foot/repeats -> missing, never normal zero. Original anchor retained with diagnostics. No total trial duration/global counts/clinical inputs/outcomes. These short-window proxies are not validated long-term variability or heel-strike PCI.',verification='All594 raw extractions independently rerun twice, same source IDs, all anchors reproduced; first12 compare render after perturbing all samples outside selected support including80msmargins. Single-cycle original19 exactly reproduced. Pilot12 before full extraction.',expected_additional_GB=.05)
    if(h/'feature-protocol.json').exists():assert json.loads((h/'feature-protocol.json').read_text())==proto
    else:save(h/'feature-protocol.json',proto)
    rows=json.loads((SRC/'cohorts.json').read_text());ids=[r['id'] for r in rows if r['status']=='ok' and r['tug'] is not None];assert len(ids)==594;original={r['id']:r for r in json.loads((SRC/'features.json').read_text())}
    if args.pilot:ids=ids[:12]
    out=[];start=time.monotonic()
    with zipfile.ZipFile(SRC/'raw/IMU_signals.zip') as z:
      mapping=filemap(z)
      for n,i in enumerate(ids):
        raw=rawread(z,mapping,i);r=one(raw,original[i]);again=one(rawread(z,mapping,i),original[i]);assert r==again;assert r['window']['support_seconds']<=4.17
        if r['audit']['cycles']==1:
            for k,v in original[i]['features'].items():assert r['features'][k]==v
        if n<12:
            w=r['window'];ref=raw['sacrum back'].PacketCounter.iloc[w['support_start']:w['support_end_exclusive']].to_numpy();alter={s:d.copy() for s,d in raw.items()}
            for d in alter.values():d.loc[~d.PacketCounter.isin(ref),d.columns!='PacketCounter']=99999.
            af=render(alter,r['audit']['selected_events'],original[i]);assert af['features']==r['features']
        out.append(dict(id=i,**r))
        if args.pilot or (n+1)%50==0:print('replayed',n+1,'feet',r['features']['observed_feet'],'cycles',r['audit']['cycles'],round(time.monotonic()-start,1),flush=True)
    name='pilot' if args.pilot else 'features';save(b/(name+'.json'),out)
    save(h/('pilot-verification.json' if args.pilot else 'feature-verification.json'),dict(status='passed',people=len(out),replayed=len(out),feature_count=len(out[0]['features']),bilateral=sum(r['features']['observed_feet']==2 for r in out),with_within_foot_repeats=sum(r['features']['within_foot_relative_std_speed'] is not None for r in out),anchor_fallbacks=sum(r['audit']['stop_reason'].startswith('original_anchor_fallback') for r in out),support_seconds_quantiles=np.quantile([r['window']['support_seconds'] for r in out],[0,.5,1]).tolist(),features_sha256=f2.sha(b/(name+'.json')),seconds=time.monotonic()-start))
if __name__=='__main__':main()
