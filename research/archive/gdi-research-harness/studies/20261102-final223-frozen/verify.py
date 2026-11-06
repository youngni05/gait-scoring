"""Independent arithmetic and identity audit; no tuning or second evaluation."""
from common import *
import csv,io
from collections import defaultdict
def main():
    locked();report=read(HERE/'results.json');lock=read(HERE/'predictions-locked.json')
    assert sha(BIG/'predictions.npz')==lock['predictions_sha256'] and sha(BIG/'scored.npz')==report['scored_sha256']
    with np.load(BIG/'scored.npz') as z:d={k:z[k].copy() for k in z.files}
    manifest=read(HERE/'models.json');people=set(map(str,d['person_id']));rebuild=[]
    for f,sf in enumerate(manifest['selections']):
        byfamily={}
        for family in ['CNN','STT']:
            parts=[]
            for m in manifest['members']:
                if m['family']!=family or m['fold']!=f:continue
                key='%s_%d_%s'%(family,f,m['tag']);parts.append(d[key])
            byfamily[family]=sum(parts)/len(parts)
        p=sf['cnn_weight']*byfamily['CNN']+(1-sf['cnn_weight'])*byfamily['STT']
        lr=p[:,0]-p[:,1];bil=d['bilateral_%d'%f];new=(1-sf['fraction'])*lr+sf['fraction']*(bil[:,0]-bil[:,1])
        correction=(new-lr)/2
        rebuild.append(np.column_stack([p[:,0]+correction,p[:,1]-correction]))
    err=float(np.abs(sum(rebuild)/5-d['fusion']).max());assert err<1e-10
    for m in manifest['members']:
        assert sha(m['checkpoint'])==m['checkpoint_sha256'];ck=torch.load(m['checkpoint'],map_location='cpu',weights_only=True)
        assert not people&set(map(str,ck['identity']['train_people'])) and not people&set(map(str,ck['identity']['validation_people']))
    sums={k:defaultdict(list) for k in ['CNN_refit','CNN_bag','fusion']}
    with (HERE/'scores.csv').open(encoding='utf-8-sig',newline='') as f:
        scored=list(csv.DictReader(f))
    assert len(scored)==report['evaluated_records']
    for r in scored:
        for k in sums:
            sums[k][r['person_id']].extend([abs(float(r[k+'_left'])-float(r['true_left'])),abs(float(r[k+'_right'])-float(r['true_right']))])
    independent={k:sum(sum(v)/len(v) for v in p.values())/len(p) for k,p in sums.items()}
    for k,v in independent.items():assert abs(v-report['metrics'][k]['person_equal_mae'])<1e-10
    # Independently re-read original labels and match every reported record's person/exam/side.
    arc=module('verify_label_archive',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    payload=arc.read_member(read(H/'studies/20261027-gdi-linkage/linkage_summary.json')['archive_member'])
    rows={r['video_id']:r for r in read(HERE/'input-membership.json')['records']};original={}
    for r in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))):
        if r['Patient_ID'] in people:original.setdefault((r['Patient_ID'],r['examid'],r['side']),[]).append(r['GDI'])
    for i,v in enumerate(d['video_id']):
        r=rows[v]
        for s,side in enumerate(['L','R']):
            values=original[r['patient_id'],r['exam_id'],side];assert len(values)==1 and float(values[0])==d['y'][i,s]
    order=sorted(people);e={k:np.array([sum(sums[k][p])/len(sums[k][p]) for p in order]) for k in sums}
    cfg=read(HERE/'protocol.json')['uncertainty'];rng=np.random.default_rng(cfg['seed']);samples=[]
    delta=e['fusion']-e['CNN_bag']
    for _ in range(cfg['draws']):samples.append(float(delta[rng.integers(0,len(order),len(order))].mean()))
    ci=np.percentile(samples,[2.5,97.5]);np.testing.assert_allclose(ci,report['comparisons']['fusion_minus_CNN_bag']['paired_person_bootstrap95'],atol=1e-12,rtol=0)
    result=dict(passed=True,protocol_sha256=sha(HERE/'protocol.json'),results_sha256=sha(HERE/'results.json'),models_verified=len(manifest['members']),reserved_vs_model_train_and_validation_overlap=0,independent_ensemble_max_difference=err,independent_person_mae=independent,independent_primary_bootstrap95=ci.tolist(),annotation_identity_and_values_verified=True,evaluated_people=len(people),evaluated_records=len(scored),post_result_training_or_tuning=False)
    save(HERE/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
