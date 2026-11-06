from study import *
def main():
    rows=json.loads((HERE/'cohort176/manifest.json').read_text());masks=np.load(HERE/'masks.npz');summary=[]
    sys.path.insert(0,str(HERE.parent.parent/'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(4,2,figsize=(13,12));ix=0
    for m in rows:
        r=m['near_chair'];summary.append({'id':m['id'],'target':m['target'],'observed_seconds':r['seconds'],'shift_seconds':r['shift_seconds'],**{f'{q}_{k}':v for q in ['main','strict'] for k,v in r[q].items() if k!='reasons'}})
        if m['id'] not in ['sub-246','sub-266','sub-310','sub-406']:continue
        a=read(long.TURN/'raw'/m['turn_file']);name=m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc');mt,mk=trc(HERE/'raw'/name);t=masks[m['id']+'_time'];mid=(t[:-1]+t[1:])/2
        root=np.column_stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_ty','pelvis_tz']]);root=uniform_filter1d(root,7,axis=0,mode='nearest')
        center=(mk['RHip']+mk['LHip'])/2;center=uniform_filter1d(np.column_stack([np.interp(t,mt,center[:,j]) for j in range(3)]),7,axis=0,mode='nearest')
        for ax in axes[ix]:
            for kind,w in r['candidates'].items():ax.axvspan(max(w[0],t[0]),min(w[1],t[-1]),color='#ffb74d',alpha=.25)
            for kind in ['departure','return']:
                for lo,hi in long.runs(masks[m['id']+'_main_'+kind]):ax.axvspan(t[lo],t[hi],color='#43a047',alpha=.45)
            ax.set(xlim=(t[0],t[-1]),xlabel='원본 시각 (초)');ax.grid(alpha=.2)
        axes[ix,0].plot(t,np.linalg.norm(np.gradient(root,1/30,axis=0)[:,[0,2]],axis=1),label='골반 모델')
        axes[ix,0].plot(t,np.linalg.norm(np.gradient(center,1/30,axis=0)[:,[0,2]],axis=1),label='원본 마커',alpha=.8)
        axes[ix,0].set(title=f'{m["id"]}: 추가 인정 {r["main"]["departure_seconds"]+r["main"]["return_seconds"]:.2f}초',ylabel='수평 속도 (m/s)')
        for p,label in [(root,'골반 모델'),(center,'원본 마커')]:axes[ix,1].plot(t,np.gradient(p[:,1],1/30),label=label)
        axes[ix,1].axhspan(-.05,.05,color='#43a047',alpha=.1);axes[ix,1].set(title='수직 움직임: 기립·착석 혼입 제한',ylabel='수직 속도 (m/s)');ix+=1
    axes[0,0].legend();axes[0,1].legend();fig.suptitle('주황: 관찰된 후보 영역 / 초록: 모든 기본 품질 기준 통과\n준비·기립·착석 시간은 입력하지 않음 · 마커도 동일 OpenCap 추정이므로 독립 정답은 아님')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(HERE/'quality-review.png',dpi=140);plt.close(fig)
    pd.DataFrame(summary).to_csv(HERE/'quality-person.csv',index=False)
    print(pd.DataFrame(summary).query('main_departure_seconds>0 or main_return_seconds>0').to_string(index=False),flush=True)
if __name__=='__main__':main()
