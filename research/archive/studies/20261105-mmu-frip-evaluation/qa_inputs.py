"""Window replay, formulas and body trajectories before any prediction fitting."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import prepare as p
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
def main():
    d=np.load(p.B/'input.npz');audit=json.loads((p.H/'input-audit.json').read_text());assert len(d['ids'])==125 and len(set(d['ids']))==125 and sum(d['age']>=65)==56
    assert p.sha(p.B/'input.npz')==audit['input_sha256'];names=list(d['feature_names']);source=json.loads((p.H/'acquisition.json').read_text());expected={r['path']:r['sha256'] for r in source['members']};maximum=0
    for i,pid in enumerate(d['ids']):
        src=p.S/'Raw/2D'/f'{pid}_2D'/f'{pid}_2D.json';assert p.sha(src)==expected[str(src)]
        frames,a=p.load_pose(pid);row=audit['records'][i];start,direction,cluster=p.select(frames,a);assert int(frames[start])==row['frame_start'] and direction==row['direction']
        win=a[start:start+60];np.testing.assert_array_equal(win,d['windows'][i]);f,seq,scale=p.transform(win,direction);np.testing.assert_array_equal(seq,d['sequence'][i]);np.testing.assert_allclose(list(f.values()),d['X'][i],rtol=0,atol=1e-12)
        xy=savgol_filter(win[:,:,:2],7,2,axis=0);hip=(xy[:,11]+xy[:,12])/2;t=np.arange(60)/30;coef=np.polyfit(t,hip[:,0],1)[0];height=np.median(abs(xy[:,18,1]-(xy[:,15,1]+xy[:,16,1])/2));speed=abs(coef)/height
        maximum=max(maximum,abs(speed-d['X'][i,0]));assert abs(speed-d['X'][i,0])<1e-10
        # A coordinate translation and a uniform pixel scale must preserve normalized inputs.
        changed=win.copy();changed[:,:,:2]=changed[:,:,:2]*1.2+[17,-9];ff,ss,_=p.transform(changed,direction)
        np.testing.assert_allclose(seq,ss,rtol=0,atol=2e-6);np.testing.assert_allclose(list(ff.values()),list(f.values()),rtol=0,atol=1e-8)
    chosen=np.linspace(0,124,12).astype(int);fig,ax=plt.subplots(12,4,figsize=(9,22));bones=[(0,18),(18,5),(18,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for row,i in enumerate(chosen):
        win=d['windows'][i];xy=savgol_filter(win[:,:,:2],7,2,axis=0);hip=xy[:,[11,12]].mean(1);scale=np.median(abs(xy[:,18,1]-xy[:,[15,16],1].mean(1)))
        xx=(xy-hip[:,None,:])/scale
        for col,fr in enumerate([0,20,40,59]):
            a=ax[row,col]
            for j,k in bones:a.plot(xx[fr,[j,k],0],-xx[fr,[j,k],1],lw=1)
            a.scatter(xx[fr,p.JOINTS,0],-xx[fr,p.JOINTS,1],s=4);a.set_xlim(-.8,.8);a.set_ylim(-.8,1.1);a.set_aspect('equal');a.axis('off');a.set_title(f'{d["ids"][i]} frame{fr}',fontsize=8)
    fig.tight_layout();fig.savefig(p.H/'pose-window-QA.png',dpi=100);plt.close(fig)
    p.save(p.H/'input-verification.json',{'status':'passed','source_people':125,'raw_track_window_replays':125,'older65':56,'independent_speed_rebuilds':125,'speed_max_difference':maximum,'translation_scale_invariance_people':125,'plot_people':12,'input_sha256':p.sha(p.B/'input.npz'),'limits':'SourceRGBnotavailable;humanidentitytrackingnotindependentlyannotated. Pixelunitsnotmetricphysicalaccuracy. No modelshavebeenfit.'})
    print('INPUT_QA_PASSED',125,'PEOPLE',maximum,flush=True)
if __name__=='__main__':main()
