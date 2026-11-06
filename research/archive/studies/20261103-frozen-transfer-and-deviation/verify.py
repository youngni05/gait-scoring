from prepare import *
from scipy.stats import rankdata

def main():
 protocol=sha(HERE/'protocol.json');prep=read(HERE/'preparation.json');nr=read(HERE/'new-model-results.json');fr=read(HERE/'frozen-verification.json');results=read(HERE/'results.json')
 assert all(r['protocol_sha256']==protocol for r in [prep,nr,fr,results])
 assert sha(BIG/'features.npz')==prep['features_sha256']==nr['features_sha256']==fr['features_sha256']
 assert sha(BIG/'new-model-oof.npz')==nr['predictions_sha256']==results['new_prediction_sha256']
 assert sha(BIG/'frozen-predictions.npz')==fr['predictions_sha256']==results['frozen_prediction_sha256']
 feature=np.load(BIG/'features.npz');new=np.load(BIG/'new-model-oof.npz');frozen=np.load(BIG/'frozen-predictions.npz');people=read(HERE/'people.json');x=feature['person_features'];curves_archive=np.load('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/person-curves.npz');curves=np.array([curves_archive[p['cohort']][p['mat_index']-1,:,:,::20] for p in people]);assert len(set(feature['person_id']))==len(people)
 details=[]
 for name,record in nr['models'].items():
  assigned=[]
  for fold in record['folds']:
   ck=np.load(fold['checkpoint']);assert sha(fold['checkpoint'])==fold['sha256'];tr=ck['train'];te=ck['test'];ref=ck['reference'];assert not set(tr)&set(te) and set(ref)<=set(tr);assigned.extend(te)
   assert all(people[i]['cohort']=='healthy' and (people[i]['age']>=65 or name=='allhealthy') for i in ref)
   for inner in fold['inner']:
    assert not set(inner['train'])&set(inner['validation']);assert set(inner['train'])|set(inner['validation'])==set(tr);assert set(inner['reference'])<=set(inner['train']);assert not set(inner['reference'])&set(te)
   assert fold['alpha']==[.1,1,10,100,1000][int(np.argmin(fold['inner_mae']))]
   independent_y=[]
   for i in tr:
    ri=[j for j in ref if j!=i];reference=np.mean([curves[j].mean(0) for j in ri],axis=0);independent_y.append(np.mean([np.linalg.norm(curves[i,k]-reference)/np.sqrt(459) for k in range(2)]))
   independent_y=np.array(independent_y);mu=x[tr].mean(0);sd=x[tr].std(0);sd[sd<1e-12]=1;z=(x[tr]-mu)/sd
   w=np.linalg.solve(z.T@z+float(ck['alpha'])*np.eye(x.shape[1]),z.T@(independent_y-independent_y.mean()))
   replay=(x[te]-ck['mean'])/ck['scale']@ck['weight']+ck['intercept'];independent=(x[te]-mu)/sd@w+independent_y.mean();err=float(np.max(np.abs(replay-independent)));assert err<1e-8
   assert np.max(np.abs(replay-new[name+'_prediction'][te]))<1e-10
   reference=np.mean([curves[j].mean(0) for j in ref],axis=0);yt=np.array([np.mean([np.linalg.norm(curves[i,k]-reference)/np.sqrt(459) for k in range(2)]) for i in te]);assert np.max(np.abs(yt-new[name+'_truth'][te]))<1e-10
   details.append(dict(reference=name,fold=fold['fold'],reference_n=len(ref),independent_closed_form_max_difference=err))
  assert sorted(assigned)==list(range(len(people)))
  mae=float(np.mean(np.abs(new[name+'_prediction']-new[name+'_truth'])));assert abs(mae-record['mae_degrees'])<1e-12
 manifest=read(ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json');parts=[]
 for f,selection in enumerate(manifest['selections']):
  family={}
  for name in ['CNN','STT']:
   keys=[f'{name}_{f}_{m["tag"]}' for m in manifest['members'] if m['family']==name and m['fold']==f];family[name]=sum((frozen[k] for k in keys))/len(keys)
  base=family['CNN']*selection['cnn_weight']+family['STT']*(1-selection['cnn_weight']);bil=frozen[f'bilateral_{f}'];mean=base.sum(axis=1)/2;difference=(base[:,0]-base[:,1])*(1-selection['fraction'])+(bil[:,0]-bil[:,1])*selection['fraction'];parts.append(np.stack([mean+difference/2,mean-difference/2],axis=1))
 fusion=sum(parts)/5;ensemble_error=float(np.max(np.abs(fusion-frozen['fusion'])));assert ensemble_error<1e-10
 assert len(fr['checks'])==54 and max(c['prior_replay_max'] for c in fr['checks'])<1e-4
 scores={s['id']:s for s in read(HERE/'scores.json')};ids=results['clinical_comparison']['people'];poma=rankdata([scores[i]['clinical']['POMA'] for i in ids]);speed=rankdata([scores[i]['clinical']['speed']['mean_mps'] for i in ids]);partial_checks=[]
 for name,values in [('fusion',[scores[i]['frozen']['fusion'] for i in ids]),('new_age65plus',[-scores[i]['new_deviation'] for i in ids])]:
  rank=rankdata(values);r=np.corrcoef([rank,poma,speed]);partial=(r[0,1]-r[0,2]*r[1,2])/np.sqrt((1-r[0,2]**2)*(1-r[1,2]**2));assert abs(partial-results['clinical_comparison']['models'][name]['POMA_partial_spearman_speed'])<1e-10;partial_checks.append(dict(model=name,partial=float(partial)))
 # Window identifiers stay within the same source trial, camera and proxy condition.
 rows=read(HERE/'rows.json');assert len(feature['owner'])==len(feature['x']) and feature['x'].shape[1:]==(2,124,12)
 for i,r in enumerate(rows):assert sorted(feature['start'][feature['owner']==i].tolist())==r['starts']
 # Clinical subgroup errors and tails avoid hiding heterogeneous performance.
 subgroups={}
 for group in ['healthy65plus','stroke65plus','all']:
  ix=np.array([i for i,p in enumerate(people) if group=='all' or (p['age']>=65 and p['cohort']==('healthy' if group=='healthy65plus' else 'stroke'))]);err=new['age65plus_prediction'][ix]-new['age65plus_truth'][ix];subgroups[group]=dict(n=len(ix),mae=float(np.abs(err).mean()),bias=float(err.mean()),p90_absolute_error=float(np.quantile(np.abs(err),.9)),max_absolute_error=float(np.max(np.abs(err))))
 report=dict(passed=True,protocol_sha256=protocol,independent_replays=details,frozen_prior_replays=54,frozen_prior_max_error=max(c['prior_replay_max'] for c in fr['checks']),ensemble_max_error=ensemble_error,partial_correlation_checks=partial_checks,subgroups=subgroups,source_archives_provider_md5_verified=True,new_model_gpu_jobs=0,clinical_validation=False,skill_validation='Both installed and project copies passed quick_validate; SHA equality checked')
 save(HERE/'verification.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
