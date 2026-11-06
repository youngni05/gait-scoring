"""Descriptive case audit. Chair signals are never gait predictors."""
from common import *
import io,re
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mot(p):
    lines=p.read_text().splitlines();n=lines.index('endheader')
    return pd.read_csv(io.StringIO('\n'.join(lines[n+1:])),sep=r'\s+')
def trc(p):
    lines=p.read_text().splitlines();meta=lines[2].split();assert meta[4]=='m'
    names=[x for x in lines[3].split('\t')[2:] if x]
    a=np.loadtxt(io.StringIO('\n'.join(lines[6:])));xyz=a[:,2:].reshape(len(a),len(names),3)
    assert len(a)==int(meta[2]) and np.isfinite(xyz).all()
    return a[:,1],dict(zip(names,xyz.transpose(1,0,2)))

def extra(x):
    # Values absent from original14: absolute levels, other planes, phase, speed ordering.
    s=uniform_filter1d(x[:17].astype(float),5,axis=1,mode='nearest')
    out={}
    for name,cs in [('knee',[5,11]),('hip_flexion',[2,8]),('hip_adduction',[3,9]),('hip_rotation',[4,10]),('ankle',[6,12])]:
        for side,c in zip(['r','l'],cs):
            out[name+'_'+side+'_median']=float(np.median(s[c]))
            out[name+'_'+side+'_p10']=float(np.percentile(s[c],10))
            if name=='knee':out[name+'_'+side+'_zero_fraction']=float(np.mean(abs(x[c])<1e-6))
    for name,c in [('pelvis_tilt',0),('lumbar_extension',14)]:out[name+'_median']=float(np.median(s[c]))
    out['speed_late_minus_early']=float(x[17,91:].mean()-x[17,:90].mean())
    out['knee_lr_correlation']=float(np.corrcoef(s[5],s[11])[0,1])
    _,info=quality.prior.patterns(x);waves=[]
    for side,channels in enumerate([[2,3,4,5,6],[8,9,10,11,12]]):
        spans=info['spans'][side]
        waves.extend([np.mean([np.interp(np.linspace(a,b,51),np.arange(181),s[c]) for a,b in spans],axis=0) if spans else np.full(51,np.nan) for c in channels])
    return out,np.array(waves)

def main():
    d=load();manifest=pd.read_csv(PRIOR/'STS_common6/manifest.csv').set_index('id').loc[d['ids']]
    labels=pd.read_csv(LABELS);assert not labels.duplicated(['subid','visit']).any()
    labels=labels.set_index(['subid','visit'])
    index=json.loads((SOURCE/'archive-index.json').read_text());pairs=pd.read_csv(HERE/'selected-pairs.csv')
    selected=set(pairs.id_a)|set(pairs.id_b);q=pd.read_csv(QUALITY/'quality-audit.csv').query("dataset=='STS_common6'").set_index('id')
    provenance=[];features=[];waveforms=[];people=[];chairs=[]
    for i,pid in enumerate(d['ids']):
        r=manifest.loc[pid];lab=labels.loc[(pid,int(r.visit))];assert abs(lab.tft_5xsts-d['y'][i])<1e-8
        assert r.seconds==6 and pid!='sub-283'
        files=[a['name'] for a in index if f'/{pid}/visit-{int(r.visit)}/' in a['name'] and a['name'].endswith('.mot') and 'xsts' in a['name']]
        tasks=[re.search(r'task-([^/]+)\.mot$',f).group(1) for f in files]
        provenance.append(dict(id=pid,visit=int(r.visit),label=float(lab.tft_5xsts),chair_tasks=';'.join(tasks),five_available=any(t.startswith('5xsts') for t in tasks),three_available=any(t.startswith('3xsts') for t in tasks),chair_files=';'.join(files)))
        f,w=extra(d['x'][i]);features.append(dict(id=pid,**f));waveforms.append(w)
        if pid not in selected:continue
        source=SOURCE/'raw'/r['file'];assert sha(source)==r.source_sha256
        a=mot(source);t=float(r.start_time)+np.arange(181)/30
        mp=HERE/'raw'/r['file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')
        mt,m=trc(mp);assert mt[0]<=t[0]+1e-6 and mt[-1]>=t[-1]-1e-6
        interp=lambda v:np.column_stack([np.interp(t,mt,v[:,j]) for j in range(3)])
        hip=(m['RHip']+m['LHip'])/2;hp=interp(hip)
        root=np.column_stack([np.interp(t,a.time,a[c]) for c in ['pelvis_tx','pelvis_ty','pelvis_tz']])
        hv=np.gradient(uniform_filter1d(hp,7,axis=0,mode='nearest'),1/30,axis=0)
        rv=np.gradient(uniform_filter1d(root,7,axis=0,mode='nearest'),1/30,axis=0)
        agreement=np.linalg.norm((hv-rv)[5:-5][:,[0,2]],axis=1)
        native=(mt[:-1]>=t[0])&(mt[1:]<=t[-1]);vel=np.diff(hip,axis=0)/np.diff(mt)[:,None]
        jump=(np.linalg.norm(vel[:,[0,2]],axis=1)>6)|(abs(vel[:,1])>3)
        widths={}
        for part,l,rgt in [('hip','LHip','RHip'),('shoulder','LShoulder','RShoulder')]:
            v=np.linalg.norm(interp(m[l])-interp(m[rgt]),axis=1);rel=abs(v/np.median(v)-1)
            widths[part+'_width_p95_relative_deviation']=float(np.percentile(rel,95))
            widths[part+'_width_over25_fraction']=float(np.mean(rel>.25))
        geometry={}
        for side in ['R','L']:
            proximal=interp(m[side+'Hip']-m[side+'Knee']);distal=interp(m[side+'Ankle']-m[side+'Knee'])
            angle=180-np.degrees(np.arccos(np.clip(np.sum(proximal*distal,axis=1)/(np.linalg.norm(proximal,axis=1)*np.linalg.norm(distal,axis=1)),-1,1)))
            angle=uniform_filter1d(angle,5,mode='nearest')
            geometry[side.lower()+'_marker_knee_p10']=float(np.percentile(angle,10))
            geometry[side.lower()+'_marker_knee_median']=float(np.median(angle))
        rec=dict(id=pid,visit=int(r.visit),walking_session=int(r.session),start_seconds=t[0],end_seconds=t[-1],label=float(lab.tft_5xsts),
                 diagnosis=lab['diag'],diagnosis_subtype=str(lab.diag_sub),age=float(lab.age),sex=lab.sex,height=float(lab.height),
                 tft_tug=float(lab.tft_tug),clinical_10mwt=float(lab.tft_10mwt),
                 stable=bool(q.loc[pid].stable),bad_angle_fraction=float(q.loc[pid].bad_fraction),cycle_spread=float(q.loc[pid].cycle_seconds_spread),
                 marker_jump_fraction=float(jump[native].mean()),velocity_disagreement_median=float(np.median(agreement)),velocity_disagreement_p95=float(np.percentile(agreement,95)),
                 **widths,**geometry,**f,mot_sha256=sha(source),trc_sha256=sha(mp),chair_tasks=';'.join(tasks))
        people.append(rec)
        for cf in files:
            p=HERE/'raw'/cf;b=mot(p);bt=b.time.to_numpy();dt=np.median(np.diff(bt));width=max(3,int(round(.15/dt))|1)
            h=uniform_filter1d(b.pelvis_ty.to_numpy(),width,mode='nearest');kn=uniform_filter1d((b.knee_angle_r+b.knee_angle_l).to_numpy()/2,width,mode='nearest')
            peaks,props=find_peaks(h,prominence=.12,distance=max(1,int(.8/dt)))
            lows,_=find_peaks(-h,prominence=.12,distance=max(1,int(.8/dt)))
            cr=dict(id=pid,file=cf,task=re.search(r'task-([^/]+)\.mot$',cf).group(1),label=float(lab.tft_5xsts),record_seconds=float(bt[-1]-bt[0]),
                    candidate_height_peaks=bt[peaks].tolist(),candidate_height_troughs=bt[lows].tolist(),height_range=float(np.ptp(h)),sha256=sha(p))
            chairs.append(cr)
            fig,ax=plt.subplots(2,1,figsize=(10,4),sharex=True)
            ax[0].plot(bt,h);ax[0].scatter(bt[peaks],h[peaks],color='red');ax[0].set_ylabel('Pelvis height (m)')
            ax[1].plot(bt,kn);ax[1].set_ylabel('Mean knee (deg)');ax[1].set_xlabel('Recording time (s)')
            fig.suptitle(f'{pid}: {cr["task"]}, clinical table 5xSTS={lab.tft_5xsts:.2f}s (audit only)')
            fig.tight_layout();fig.savefig(HERE/f'chair-{pid}.png',dpi=140);plt.close(fig)
    prov=pd.DataFrame(provenance);prov.to_csv(HERE/'label-provenance-all.csv',index=False)
    pd.DataFrame(people).to_csv(HERE/'person-audit.csv',index=False);save('chair-audit.json',chairs)
    f=pd.DataFrame(features).set_index('id');f.to_csv(HERE/'extra-features-all.csv')
    scale=(f.quantile(.75)-f.quantile(.25)).clip(lower=1e-8)
    differences=[];wave_metrics=[];waves=np.array(waveforms);np.savez_compressed(HERE/'cycle-waveforms.npz',ids=d['ids'],waves=waves)
    for row in pairs.itertuples():
        i,j=int(row.i),int(row.j);diff=abs(f.iloc[i]-f.iloc[j]);norm=diff/scale
        for name in f.columns:differences.append(dict(case=row.case,id_a=row.id_a,id_b=row.id_b,feature=name,a=f.iloc[i][name],b=f.iloc[j][name],absolute_difference=diff[name],difference_in_cohort_iqr=norm[name]))
        fig,ax=plt.subplots(2,3,figsize=(12,6));cols=[5,2,14,11,8,0];names=['Right knee','Right hip flexion','Lumbar extension','Left knee','Left hip flexion','Pelvis tilt']
        for c,n,axis in zip(cols,names,ax.ravel()):
            for k,color in [(i,'#2764ad'),(j,'#d57427')]:axis.plot(np.arange(181)/30,d['x'][k,c],color=color,label=f'{d["ids"][k]} / {d["y"][k]:.2f}s')
            axis.set_title(n);axis.set_xlabel('Same 6s input (s)');axis.set_ylabel('deg')
        ax[0,0].legend(fontsize=8);fig.suptitle(f'{row.case}: similar summaries, unaligned raw angle traces');fig.tight_layout();fig.savefig(HERE/f'gait-{row.case}.png',dpi=150);plt.close(fig)
        # Align each side to its own knee flexion peak. Shape-only removes each curve mean.
        delta=waves[i]-waves[j];shape=delta-np.mean(delta,axis=1,keepdims=True)
        for c,name in enumerate(['r_hip_flexion','r_hip_adduction','r_hip_rotation','r_knee','r_ankle','l_hip_flexion','l_hip_adduction','l_hip_rotation','l_knee','l_ankle']):
            wave_metrics.append(dict(case=row.case,channel=name,rmse=float(np.sqrt(np.mean(delta[c]**2))),shape_only_rmse=float(np.sqrt(np.mean(shape[c]**2))),mean_offset=float(np.mean(delta[c]))))
        fig,ax=plt.subplots(2,3,figsize=(11,5));names=['hip flexion','knee','ankle'];channels=[0,3,4,5,8,9]
        for c,axis,name in zip(channels,ax.ravel(),['Right '+x for x in names]+['Left '+x for x in names]):
            for k,color in [(i,'#2764ad'),(j,'#d57427')]:axis.plot(np.linspace(0,100,51),waves[k,c],color=color,label=d['ids'][k])
            axis.set_title(f'{name}: shape RMSE {np.sqrt(np.mean(shape[c]**2)):.1f} deg');axis.set_xlabel('Knee-peak cycle (%)');axis.set_ylabel('deg')
        ax[0,0].legend();fig.tight_layout();fig.savefig(HERE/f'cycles-{row.case}.png',dpi=140);plt.close(fig)
    pd.DataFrame(differences).sort_values(['case','difference_in_cohort_iqr'],ascending=[True,False]).to_csv(HERE/'pair-extra-differences.csv',index=False)
    pd.DataFrame(wave_metrics).to_csv(HERE/'pair-waveform-differences.csv',index=False)
    allp=pd.read_csv(HERE/'all-pairs.csv');hasfive=set(prov.loc[prov.five_available,'id'])
    remaining=allp[allp.primary & allp.discordant & allp.id_a.isin(hasfive) & allp.id_b.isin(hasfive)]
    summary={'people':len(prov),'five_task_available':int(prov.five_available.sum()),'three_task_available':int(prov.three_available.sum()),
             'only_three':prov[prov.three_available&~prov.five_available].id.tolist(),'no_chair_task':prov[prov.chair_tasks==''].id.tolist(),
             'other_repetition_tasks':prov[~prov.five_available&(prov.chair_tasks!='')][['id','chair_tasks']].to_dict('records'),
             'selected_label_longer_than_record':[c['id'] for c in chairs if c['label']>c['record_seconds']],
             'primary_large_gap_with_five_task_both':len(remaining),'selected_people':len(people),
             'source_rgb_files':sum(Path(x['name']).suffix.lower() in ['.mp4','.mov','.avi'] for x in index),
             'limits':['Same-visit label match does not prove exact clinical trial correspondence.','3xsts naming is a protocol flag, not proof the clinical5xSTS value is wrong.','MOT/TRC consistency is not independent tracking ground truth.','Posthoc extra-feature differences are not held-out predictive gains.']}
    save('audit-summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
