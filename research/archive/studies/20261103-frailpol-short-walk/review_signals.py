from extract import *
def main():
    records=json.loads((BIG/'features.json').read_text(encoding='utf8'));checks=[];plots=[]
    with zipfile.ZipFile(BIG/'raw/IMU_signals.zip') as z:
        mapping=filemap(z)
        for i,r in enumerate(records):
            if r['status']!='ok':continue
            raw=rawread(z,mapping,r['id']);f,w=calculate(raw,r['selected_events']);names=list(r['features']);delta=max(abs(f[k]-r['features'][k]) for k in names)
            assert delta<1e-10 and w==r['window'];assert .6<=w['support_seconds']<=3.17
            h,_=heading(raw);assert np.ptp(h[w['core_start']:w['core_end']])<=20
            for s,e in r['selected_events'].items():
                assert abs(f['speed']-f['stride_length']/f['stride_time'])<1e-12
                assert e['start']<e['tc']<e['ic']<e['end']
            perturb=None
            if len(checks)<12 or r['id'] in PILOT:
                altered={s:d.copy() for s,d in raw.items()}
                for d in altered.values():
                    d.iloc[:w['support_start'],1:]=123456.;d.iloc[w['support_end_exclusive']:,1:]=-123456.
                af,_=calculate(altered,r['selected_events']);perturb=max(abs(af[k]-f[k]) for k in names);assert perturb==0
            checks.append(dict(id=r['id'],max_abs_replay_difference=delta,outside_window_perturbation_difference=perturb))
            if r['id'] in PILOT:plots.append((r,raw))
            if len(checks)%100==0:print('signal replay',len(checks),flush=True)
    fig,axs=plt.subplots(5,2,figsize=(13,12));
    for ax,(r,raw) in zip(axs.flat,plots):
        b=convert_to_fbf(fsf(raw),left_like='left',right_like='right');w=r['window'];lo=max(0,w['core_start']-100);hi=min(len(raw['sacrum back']),w['core_end']+100)
        for s in b:ax.plot(np.arange(lo,hi)/100,b[s].gyr_ml.iloc[lo:hi],lw=.8,label=s)
        ax.axvspan(w['core_start']/100,w['core_end']/100,color='green',alpha=.15);ax.set_title(f"ID {r['id']} | support {w['support_seconds']:.2f}s | speed {r['features']['speed']:.2f}m/s");ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'selected-window-review.png',dpi=120);plt.close(fig)
    save(HERE/'signal-verification.json',dict(status='passed',people=len(checks),max_abs_difference=max(r['max_abs_replay_difference'] for r in checks),perturbation_people=sum(r['outside_window_perturbation_difference'] is not None for r in checks),checks=checks,code_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
