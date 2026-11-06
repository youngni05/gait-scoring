from study import *
def main():
    proto=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==proto['code_sha256']
    for f,h in proto['source_hashes'].items():assert sha(HERE.parent/f)==h
    info=proto['cohorts']['cohort176'];out=HERE/'cohort176';data=np.load(out/'input.npz');prior=np.load(PRIOR/'exclude283/input.npz');masks=np.load(HERE/'masks.npz')
    for f,h in info['hashes'].items():assert sha(out/f)==h
    assert np.array_equal(data['Original'],prior['Baseline_20']) and np.array_equal(data['ids'],prior['ids']) and np.array_equal(data['y'],prior['y'])
    assert len(set(data['ids']))==176 and 'sub-283' not in data['ids'];assert sha(out/'splits.json')==sha(PRIOR/'exclude283/splits.json')
    scouts={r['id']:r for r in json.loads((long.HERE/'scout.json').read_text())};phases={r['id']:r['phases'] for r in json.loads((HERE.parent/'20261031-opencap-tail-diagnosis/person-audit.json').read_text())}
    rows=json.loads((out/'manifest.json').read_text());maximum=0.;sources=[]
    for i,m in enumerate(rows):
        a=read(long.TURN/'raw'/m['turn_file']);assert sha(long.TURN/'raw'/m['turn_file'])==m['turn_sha256'];sources.append(m['turn_sha256'])
        name=m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc');marker=trc(HERE/'raw'/name) if (HERE/'raw'/name).exists() else None
        views,r,mm=features(a,m,scouts[m['id']],phases[m['id']],marker);assert r==m['near_chair']
        assert abs(r['seconds']-m['budgets']['20']['observed_seconds'])<1e-8 and r['seconds']<=20+1e-8
        assert r['window'][0]>=a['time'][0]-1e-8 and r['window'][1]<=a['time'][-1]+1e-6
        for w in m['tug_walk_windows']+[[m['turn_start'],m['turn_end']]]:assert r['window'][0]<=w[0]+1e-6 and r['window'][1]>=w[1]-1e-6
        for k,v in views.items():
            diff=float(max(abs(v-data[k][i])));maximum=max(diff,maximum);assert diff<1e-12
        for k,v in mm.items():assert np.array_equal(v,masks[f'{m["id"]}_{k}'])
        for q in ['main','strict']:
            dep=mm[q+'_departure'];ret=mm[q+'_return'];rot=mm[q+'_rotation'];assert not (dep&ret).any() and not ((dep|ret)&mm['valid']).any() and not (rot&~ret).any()
            for kind in ['departure','return','rotation']:assert mm[q+'_'+kind].sum()/30==r[q][kind+'_seconds']
        assert not ((mm['strict_departure']|mm['strict_return'])&~(mm['main_departure']|mm['main_return'])).any()
        assert np.allclose(data['AddBoth'][i,:-1],data['Matched'][i,:-1])
        expected=data['Matched'][i,-1]+r['main']['departure_seconds']+r['main']['return_seconds'];assert abs(data['AddBoth'][i,-1]-expected)<1e-12
    for rep in json.loads((out/'splits.json').read_text()):
        tests=[]
        for f in rep['splits']:
            tr=set(f['train']);te=set(f['test']);assert not tr&te and tr|te==set(range(176));tests+=f['test'];inner=[]
            for v in f['inner']:
                assert not set(v['train'])&set(v['validation']) and set(v['train'])|set(v['validation'])==tr;inner+=v['validation']
            assert sorted(inner)==sorted(tr)
        assert sorted(tests)==list(range(176))
    save('input-verification.json',{'passed':True,'raw_windows_recomputed':176,'max_feature_difference':maximum,'original_baseline_exact':True,'same_person_split_and_targets':True,'per_person_observation_duration_matched':True,'no_added_overlap_with_gait_or_far_turn':True,'strict_is_subset':True,'source_sha256':sources})
    print('INPUT_VERIFIED 176 people, same time and splits',flush=True)
if __name__=='__main__':main()
