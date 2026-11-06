from study import *
def independent_episode(v,valid):
    mask=(v&valid).copy();i=0
    while i<len(mask):
        if mask[i]:i+=1;continue
        j=i
        while j<len(mask) and not mask[j]:j+=1
        if i>0 and j<len(mask) and j-i<=6 and all(valid[i:j]):mask[i:j]=True
        i=j
    lengths=[];i=0
    while i<len(mask):
        if not mask[i]:i+=1;continue
        j=i
        while j<len(mask) and mask[j]:j+=1
        if j-i>=15:lengths.append(j-i)
        i=j
    return [sum(lengths)/30,max([0]+lengths)/30]
def main():
    p=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==p['code_sha256']
    for n,h in p['source_hashes'].items():assert sha(HERE.parent/n)==h
    scouts={r['id']:r for r in json.loads((HERE/'scout.json').read_text())};maxdiff=0.
    olddata=np.load(PREV/'common176/input.npz');plans=json.loads((PREV/'common176/splits.json').read_text())
    for cohort,info in p['cohorts'].items():
        out=HERE/cohort;d=np.load(out/'input.npz');rows=json.loads((out/'manifest.json').read_text());ix=info['source_indices']
        assert np.array_equal(d['ids'],olddata['ids'][ix]) and np.array_equal(d['y'],olddata['y'][ix]) and 'sub-283' not in d['ids'] and sum(d['y']>=20)==8
        for v in ['Current','SameTUG']:assert np.array_equal(d[v],olddata[v][ix])
        for rep,oldrep in zip(json.loads((out/'splits.json').read_text()),plans):
            seen=[]
            for f,of in zip(rep['splits'],oldrep['splits']):
                assert [ix[j] for j in f['train']]==[j for j in of['train'] if j in ix]
                assert [ix[j] for j in f['test']]==[j for j in of['test'] if j in ix]
                tr,te=set(f['train']),set(f['test']);assert not tr&te and tr|te==set(range(len(ix)));seen+=f['test'];valid=[]
                for v,ov in zip(f['inner'],of['inner']):
                    assert [ix[j] for j in v['train']]==[j for j in ov['train'] if j in ix]
                    assert [ix[j] for j in v['validation']]==[j for j in ov['validation'] if j in ix]
                    it,iv=set(v['train']),set(v['validation']);assert not it&iv and it|iv==tr;valid+=v['validation']
                assert sorted(valid)==sorted(tr)
            assert sorted(seen)==list(range(len(ix)))
        for i,m in enumerate(rows):
            a=read(TURN/'raw'/m['turn_file']);s=scouts[m['id']];assert sha(TURN/'raw'/m['turn_file'])==m['turn_sha256'];previous=None
            for b in [10,20,30]:
                r=m['budgets'][str(b)];l,h=r['window'];t=l+np.arange(round((h-l)*30)+1)/30;mid=(t[1:]+t[:-1])/2
                assert len(t)-1==round(r['observed_seconds']*30) and h<=a['time'][-1]+1e-6 and l>=s['start']-1e-6 and h-l<=b+1e-6
                assert l<=m['turn_start'] and h>=m['turn_end']
                if previous:assert l<=previous[0]+1e-6 and h>=previous[1]-1e-6
                previous=[l,h]
                if cohort=='exact10' and b==10:assert abs(h-l-10)<1e-6
                pos=np.stack([np.interp(t,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1);speed=np.linalg.norm(np.gradient(pos,1/30,axis=0),axis=1)
                sm=np.stack([np.convolve(np.pad(pos[:,j],3,mode='edge'),np.ones(7)/7,mode='valid') for j in range(2)],axis=1);vv=np.gradient(sm,1/30,axis=0)
                horizontal=np.linalg.norm(vv,axis=1)[:-1];axial=abs(vv@np.array(m['axis']))[:-1]
                valid=np.array([any(w and w[0]-1e-6<=x<=w[1]+1e-6 for w in [s['pre'],s['post']]) for x in mid])
                initial=np.median(a['pelvis_ty'][a['time']<=a['time'][0]+.5])
                seated=(np.interp(mid,a['time'],a['pelvis_ty'])<initial+.12)&(np.interp(mid,a['time'],(a['knee_angle_l']+a['knee_angle_r'])/2)>60)&(np.interp(mid,a['time'],(a['hip_flexion_l']+a['hip_flexion_r'])/2)>30)
                pp=np.stack([a['pelvis_tx'],a['pelvis_tz']],axis=1);native=np.sqrt(np.square(np.diff(pp,axis=0)).sum(1))/np.diff(a['time']);bad=np.zeros(len(mid),bool)
                for j in np.flatnonzero(native>6):bad|=(mid>=a['time'][j]-.15)&(mid<=a['time'][j+1]+.15)
                valid&=~seated&~bad;assert abs(valid.sum()/30-r['valid_nonturn_seconds'])<1e-9
                vals=speed[:-1][valid];base=[vals.mean(),vals.std(),abs(np.diff(speed[:-1]))[valid[:-1]&valid[1:]].mean(),m['turn_observed_seconds']]
                delta=max(abs(np.array(base)-d[f'Speed_{b}'][i]));assert delta<1e-9;maxdiff=max(maxdiff,float(delta))
                for suffix,threshold in [('10',.1),('15',.15)]:
                    durations=independent_episode(axial<threshold,valid)+independent_episode(horizontal<threshold,valid)
                    assert np.array_equal(d[f'Delay{suffix}_{b}'][i,:4],d[f'Speed_{b}'][i])
                    assert np.allclose(durations,d[f'Delay{suffix}_{b}'][i,4:],atol=1e-10,rtol=0)
        for n,h in info['hashes'].items():assert sha(out/n)==h
    save('input-verification.json',{'passed':True,'same_person_splits':True,'no_sub283':True,'nested_real_windows_and_turn_containment':True,'independent_speed_max_difference':maxdiff,'episode_durations_independently_reconstructed':True,'no_padding':True})
    print('INPUT VERIFIED',maxdiff,flush=True)
if __name__=='__main__':main()
