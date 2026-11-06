"""Frozen, outcome-blind contiguous straight-walk extension with original fallback."""
from pathlib import Path
import sys,json,zipfile,warnings,argparse,time
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
sys.path.insert(0,str(H))
import features2 as f2
from extract import heading,calculate,fsf,convert_to_fbf,BarthDtw,BarthOriginalTemplate,HerzerEventDetection,np
from signal_pilot import filemap,rawread,save
B=f2.B;SRC=f2.SRC

def extend(raw,original):
    anchor=original['selected_events'];side=next(iter(anchor));e=anchor[side]
    a=fsf(raw);bf=convert_to_fbf(a,left_like='left',right_like='right')
    with warnings.catch_warnings(record=True):
        dtw=BarthDtw(template=BarthOriginalTemplate(use_cols=['gyr_ml'])).segment(bf,sampling_rate_hz=100)
        ed=HerzerEventDetection(min_vel_search_win_size_ms=130,mid_swing_peak_prominence=.03*180/np.pi,mid_swing_n_considered_peaks=2).detect(bf,stride_list=dtw.stride_list_,sampling_rate_hz=100)
    events=ed.min_vel_event_list_[side].reset_index().to_dict('records')
    found=[q for q in events if q['start']==e['start'] and q['end']==e['end']]
    assert len(found)==1,'original_anchor_not_reproduced'
    for k in ['start','end','tc','ic']:assert found[0][k]==e[k]
    accepted=[e];vectors=[original['features']];reason='no_contiguous_next_cycle';h,horiz=heading(raw)
    while True:
        nxt=[q for q in events if q['start']==accepted[-1]['end'] and q['end']>q['start']]
        if not nxt:break
        q=sorted(nxt,key=lambda v:(v['end'],v['s_id']))[0];lo=int(e['start']);hi=int(q['end'])
        if not(q['start']<q['tc']<q['ic']<q['end'] and .6<=(q['end']-q['start'])/100<=3):reason='invalid_next_events';break
        if hi-lo>400 or hi+9>len(h):reason='short_window_cap';break
        if np.ptp(h[lo:hi])>20 or np.min(horiz[lo:hi])<.5:reason='turn_or_orientation';break
        try:v,_=calculate(raw,{side:q})
        except Exception as exc:reason='next_trajectory_failed:'+str(exc);break
        accepted.append(q);vectors.append(v)
    window=dict(original['window']);window.update(core_end=int(accepted[-1]['end']),support_end_exclusive=int(accepted[-1]['end'])+9)
    window['support_seconds']=(window['support_end_exclusive']-window['support_start'])/100
    # Average complete-cycle values; a single cycle is numerically identical to original.
    base={k:float(np.mean([v[k] for v in vectors])) for k in vectors[0]}
    return base,window,dict(cycles=len(accepted),selected_events={side:accepted},stop_reason=reason)

def one(raw,original):
    try:base,window,audit=extend(raw,original)
    except Exception as exc:
        base=original['features'];window=original['window'];audit=dict(cycles=1,stop_reason='original_fallback:'+str(exc),selected_events=original['selected_events'])
    extra,bad=f2.features(raw,window)
    return dict(features={**base,**extra},window=window,audit={**audit,'invalid_optional_sensors':bad})

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    h=H/'round03';b=B/'round03';h.mkdir(exist_ok=True);b.mkdir(exist_ok=True)
    proto=dict(code_sha256=f2.sha(Path(__file__)),library_sha256=f2.sha(H/'features2.py'),original_extractor_sha256=f2.sha(Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/extract.py')),source_sha256=f2.sha(SRC/'features.json'),cohort_sha256=f2.sha(SRC/'cohorts.json'),raw_sha256=f2.sha(SRC/'raw/IMU_signals.zip'),design='Same594 IDs and original earliest straight-cycle anchor. Extend only immediately contiguous same-foot complete strides forward, <=4s core, trunk heading range<=20deg and horizontal projection>=0.5. Stop on first failure. Same trajectory plausibility gates. Original-window fallback retains everyone. Average19 original complete-cycle features; recompute168 sensor descriptors in extended core. No total record duration, full TUG counts, window location, label or clinical inputs. Short-window statistics are not validated long-term variability.',verification='All594 replay raw extraction twice; verify original anchor; first12 feature descriptors invariant to outside-core mutation. Original19 fallback exact. Pilot12 before full extraction; fixed rules independent of outcomes.',expected_additional_GB=.05)
    if (h/'feature-protocol.json').exists():assert json.loads((h/'feature-protocol.json').read_text())==proto
    else:save(h/'feature-protocol.json',proto)
    cohort=json.loads((SRC/'cohorts.json').read_text());ids=[r['id'] for r in cohort if r['status']=='ok' and r['tug'] is not None];assert len(ids)==594
    original={r['id']:r for r in json.loads((SRC/'features.json').read_text())}
    if args.pilot:ids=ids[:12]
    out=[];start=time.monotonic()
    with zipfile.ZipFile(SRC/'raw/IMU_signals.zip') as z:
        mapping=filemap(z)
        for n,i in enumerate(ids):
            raw=rawread(z,mapping,i);r=one(raw,original[i]);again=one(rawread(z,mapping,i),original[i]);assert r==again
            if r['audit']['cycles']==1:assert all(r['features'][k]==v for k,v in original[i]['features'].items())
            assert r['window']['support_seconds']<=4.17
            if n<12:
                w=r['window'];ref=raw['sacrum back'].PacketCounter.iloc[w['core_start']:w['core_end']].to_numpy();alter={s:d.copy() for s,d in raw.items()}
                for d in alter.values():d.loc[~d.PacketCounter.isin(ref),d.columns!='PacketCounter']=99999.
                a,_=f2.features(alter,w);assert all(r['features'][k]==v for k,v in a.items())
            out.append(dict(id=i,**r))
            if args.pilot or (n+1)%50==0:print('replayed',n+1,'cycles',r['audit']['cycles'],'seconds',round(time.monotonic()-start,1),flush=True)
    name='pilot' if args.pilot else 'features';save(b/(name+'.json'),out)
    counts={str(k):sum(r['audit']['cycles']==k for r in out) for k in sorted(set(r['audit']['cycles'] for r in out))}
    save(h/('pilot-verification.json' if args.pilot else 'feature-verification.json'),dict(status='passed',people=len(out),replayed=len(out),feature_count=len(out[0]['features']),cycles=counts,anchor_fallbacks=sum(r['audit']['stop_reason'].startswith('original_fallback') for r in out),support_seconds_quantiles=np.quantile([r['window']['support_seconds'] for r in out],[0,.5,1]).tolist(),features_sha256=f2.sha(b/(name+'.json')),seconds=time.monotonic()-start))
if __name__=='__main__':main()
