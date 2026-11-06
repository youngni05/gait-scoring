"""Input-only amendment before model fitting: reject likely seated windows."""
from study import *
def clean_select(a,m):
    initial=float(np.median(a['pelvis_ty'][a['time']<=a['time'][0]+.5]));candidates=[];rejected=0
    for leg in scout.eligible(a,m)[:2]:
        if leg is None:continue
        t,pos=grid(a,[leg[0],leg[0]+np.floor((leg[1]-leg[0])*30+1e-5)/30]);_,ax=motion(pos,m['axis'])
        for j in range(0,len(t)-45,3):
            w=[float(t[j]),float(t[j]+1.5)];mask=(a['time']>=w[0])&(a['time']<=w[1])
            height=float(np.median(a['pelvis_ty'][mask]));knee=float(np.median((a['knee_angle_l'][mask]+a['knee_angle_r'][mask])/2));hip=float(np.median((a['hip_flexion_l'][mask]+a['hip_flexion_r'][mask])/2))
            seated=height<initial+.12 and knee>60 and hip>30
            if seated:rejected+=1
            else:candidates.append((float(np.mean(ax[j:j+45])),w))
    if not candidates:return m['tug_walk_windows'],{'rejected_candidates':rejected,'fallback':'no_nonseated_candidate'}
    candidate=min(candidates,key=lambda x:(x[0],x[1][0]))[1];windows=[candidate];cells=[]
    for l,r in m['tug_walk_windows']:
        for j in range(round((r-l)*30)):
            cell=[l+j/30,l+(j+1)/30]
            if overlaps(cell,candidate)<1e-6:cells.append(cell)
    assert len(cells)>=45
    for l,r in cells[:45]:
        if len(windows)>1 and abs(windows[-1][1]-l)<1e-6:windows[-1][1]=r
        else:windows.append([l,r])
    qc=features(a,windows,m['axis'])[2]
    fail=qc['max_native_horizontal_speed']>6 or qc['max_identical_body_seconds']>=.25
    return (m['tug_walk_windows'] if fail else windows),{'rejected_candidates':rejected,'fallback':'tracking_flag' if fail else None,'initial_pelvis_height':initial}

def main():
    out=HERE/'common176';assert not (out/'completion.json').exists() and not (HERE/'protocol-initial.json').exists()
    protocol=json.loads((HERE/'protocol.json').read_text());save('protocol-initial.json',protocol)
    d=dict(np.load(out/'input.npz'));rows=json.loads((out/'manifest.json').read_text());clean=[];delay=[]
    for i,m in enumerate(rows):
        a=read(TURN/'raw'/m['turn_file']);windows,qc=clean_select(a,m);s,f,_=features(a,windows,m['axis'])
        base=np.r_[s,d['SameTUG'][i,3]];clean.append(base);delay.append(np.r_[base,f])
        m.update({'clean_windows':windows,'posture_qc':qc,'clean_delay_seconds':f.tolist()})
    d.update(CleanSpeed=np.array(clean),CleanDelay=np.array(delay));np.savez_compressed(out/'input.npz',**d);save('common176/manifest.json',rows)
    protocol['cohorts']['common176']['views']+=['CleanSpeed','CleanDelay']
    protocol['cohorts']['common176']['hashes']={n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}
    protocol['input_only_amendment']={'reason':'TRC review found sub268 candidate has both knees~95deg and seated-like height; prevent returned-to-chair stillness being interpreted as walking pause. Added before any model fit or new performance inspection.',
      'gate':'reject candidate if median pelvis height < first0.5s median+0.12m AND mean bilateral knee flexion median>60deg AND mean bilateral hip flexion median>30deg; same rule applied to all176; heuristic, not validated posture classifier',
      'primary':'CleanDelay minus CleanSpeed; CleanDelay minus SameTUG; raw Delay retained as sensitivity to phase contamination',
      'fallback':'if no nonseated1.5s candidate or speed/freeze QC fail, use original3s, all people retained','code_sha256':sha(HERE/'quality_prepare.py')}
    save('protocol.json',protocol)
    changed=[m['id'] for m in rows if m['clean_windows']!=m['observed_windows']]
    print(json.dumps({'posture_filter_changed_people':changed,'fallbacks':[{m['id']:m['posture_qc']} for m in rows if m['posture_qc']['fallback']], 'long_clean_features':[{k:m[k] for k in ['id','clean_delay_seconds']} for m in rows if m['target']>=20]}),flush=True)
if __name__=='__main__':main()
