from analyze import *
from scipy.ndimage import uniform_filter1d

def main():
    rows=json.loads((HERE/'person-audit.json').read_text());manifest={m['id']:m for m in json.loads((PRIOR/'include283/manifest.json').read_text())}
    selected=[r for r in rows if r['large_under'] or r['id'] in ['sub-267','sub-283']];gaprows=[];traces=[]
    for r in selected:
        m=manifest[r['id']];a=read(source.long.TURN/'raw'/r['source_file']);t=a['time'];dt=float(np.median(np.diff(t)))
        pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);smooth=uniform_filter1d(pos,max(3,round(.25/dt)),axis=0,mode='nearest')
        vel=np.gradient(smooth,t,axis=0);speed=np.linalg.norm(vel,axis=1);h=uniform_filter1d(a['pelvis_ty'],max(3,round(.25/dt)),mode='nearest')
        p=r['phases'];gaps=[('departure',[p['rise']['window'][1],r['pre'][0]]),('return',[r['post'][1],p['sit']['window'][0]])]
        for kind,window in gaps:
            if window[1]<=window[0]:continue
            sel=(t>=window[0])&(t<=window[1]);q=pos[sel];tt=t[sel];rawspeed=np.linalg.norm(np.diff(q,axis=0),axis=1)/np.diff(tt)
            observed=m['budgets']['20']['window'];overlap=max(0,min(window[1],observed[1])-max(window[0],observed[0]))
            vertical=abs(np.diff(a['pelvis_ty'][sel])/np.diff(tt));bad=np.zeros(len(tt)-1,bool);mid=(tt[:-1]+tt[1:])/2
            for j in np.flatnonzero((rawspeed>6)|(vertical>3)):bad|=(mid>=tt[j]-.15)&(mid<=tt[j+1]+.15)
            masked=float(np.diff(tt)[bad].sum());clean_slow=float(np.diff(tt)[(~bad)&(speed[sel][:-1]<.15)].sum())
            gaprows.append({'id':r['id'],'kind':kind,'start':window[0],'end':window[1],'seconds':window[1]-window[0],'inside_observation20':overlap,'outside_observation20':window[1]-window[0]-overlap,
              'smoothed_speed_median':float(np.median(speed[sel])),'smoothed_speed_mean':float(np.mean(speed[sel])),'fraction_speed_below_010':float(np.mean(speed[sel]<.10)),
              'net_displacement':float(np.linalg.norm(q[-1]-q[0])),'native_horizontal_speed_max':float(max(rawspeed)),
              'native_vertical_speed_max':float(max(vertical)),'jump_masked_seconds':masked,'clean_below015_seconds':clean_slow,'phase_quality':r['phase_quality']})
        traces.append((r,m,t,speed,h,gaps))
    pd.DataFrame(gaprows).to_csv(HERE/'gap-review.csv',index=False)
    sys.path.insert(0,str(HERE.parent.parent/'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(6,2,figsize=(13,17))
    for axesrow,(r,m,t,speed,h,gaps) in zip(axes,traces):
        bounds=[r['phases']['rise']['window'][0]-1,r['phases']['sit']['window'][1]+1]
        for ax,v,label in zip(axesrow,[speed,h],['수평 속도 (m/s)','골반 높이 (m)']):
            ax.plot(t,v,lw=1,color='#263238')
            for window in [r['pre'],[m['turn_start'],m['turn_end']],r['post']]:ax.axvspan(*window,color='#2196f3',alpha=.12)
            for kind,window in gaps:
                if window[1]>window[0]:ax.axvspan(*window,color='#f59e0b',alpha=.4)
            for key in ['rise','sit']:
                for x in r['phases'][key]['window']:ax.axvline(x,color='#888888',ls=':',lw=.8)
            ax.set(xlim=bounds,ylabel=label,xlabel='원본 기록 시각 (초)');ax.grid(alpha=.2)
        axesrow[0].set_title(f'{r["id"]}: 실제 {r["target"]:.2f}s / 예측 {r["Baseline_prediction"]:.2f}s')
        own=[g for g in gaprows if g['id']==r['id']];jump=any(g['jump_masked_seconds']>0 for g in own)
        axesrow[1].set_title('경계·추가 구간 품질: '+('추적 급변 경고' if jump or not r['phase_quality'] else '급변 기준 통과'))
    fig.suptitle('파랑: 기존 보행·회전 영역 / 주황: 일어서기 종료~앉기 시작 사이에서 빠진 영역\n자동 경계 진단이며 실제 계측 시작·임상 정지 확인이 아님',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(HERE/'boundary-review.png',dpi=130);plt.close(fig)
    # Available expansion candidates: file presence is not acquisition or usable quality.
    idx=json.loads((HERE.parent/'20261030-opencap415-tug-timeseries/archive-index.json').read_text())
    candidates=[('sub-092',0),('sub-261',839),('sub-267',350),('sub-283',0)];audit=[]
    for pid,visit in candidates:
        files=[r['name'] for r in idx if f'/{pid}/visit-{visit}/' in r['name'] and '/Kinematics/' in r['name'] and 'task-tug' in r['name'] and r['name'].endswith('.mot')]
        audit.append({'id':pid,'visit':visit,'listed_tug_files':files,'local_turn_files':sum((source.long.TURN/'raw'/f).exists() for f in files),'status':'listing_only; eligibility and linkage not established'})
    save('expansion-candidates.json',audit)
    print(pd.DataFrame(gaprows).to_string(index=False),flush=True)
if __name__=='__main__':main()
