"""Apply the user's label correction to existing scores, without model inference."""
import json,csv,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
NORMAL=[1,2,3,6,8,9];ABNORMAL=[4,5,7,10,11,12,13]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    archive=HERE/'before-label-correction';assert not archive.exists(),'Correction already applied; inspect existing state'
    archive.mkdir()
    for name in ['labels.json','results.json','verification.json','video-scores.csv','README.md','video-scores.png']:
        shutil.copy2(HERE/name,archive/name)
    r=read(HERE/'results.json');old=read(archive/'results.json');labels=read(HERE/'labels.json')
    labels.update(source='User correction after viewing predictions: 6,8,9 are short normal-walking recordings',normal_numbers=NORMAL,abnormal_numbers=ABNORMAL,corrected_after_predictions=True,previous_labels='before-label-correction/labels.json')
    (HERE/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf8')
    for row in r['records']:row['condition']='normal' if row['number'] in NORMAL else 'abnormal'
    n=[v['score'] for v in r['records'] if v['condition']=='normal'];a=[v['score'] for v in r['records'] if v['condition']=='abnormal']
    g=dict(unit='video, no same-person matching or independence-based inference',normal_n=len(n),abnormal_n=len(a),normal_mean=sum(n)/len(n),abnormal_mean=sum(a)/len(a),normal_range=[min(n),max(n)],abnormal_range=[min(a),max(a)])
    g.update(normal_minus_abnormal=g['normal_mean']-g['abnormal_mean'],minimum_between_group_gap=min(n)-max(a),abnormal_above_lowest_normal_numbers=[v['number'] for v in r['records'] if v['condition']=='abnormal' and v['score']>=min(n)])
    r['group']=g;r['label_correction']=dict(changed_numbers=[6,8,9],previous_results='before-label-correction/results.json',after_predictions=True,model_inference_repeated=False)
    (HERE/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    fields=['number','condition','score','left','right','valid_windows','window_min','window_max','fold_min','fold_max','status']
    with (HERE/'video-scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(r['records'])
    for before,after in zip(old['records'],r['records']):
        assert {k:v for k,v in before.items() if k!='condition'}=={k:v for k,v in after.items() if k!='condition'}
    assert old['predictions_sha256']==r['predictions_sha256'] and old['features_sha256']==r['features_sha256']
    proof=dict(changed_numbers=[6,8,9],source='User correction in this chat',corrected_after_predictions=True,all_numeric_record_fields_unchanged=True,predictions_sha256_unchanged=r['predictions_sha256'],previous_results_sha256=sha(archive/'results.json'),current_results_sha256=sha(HERE/'results.json'),model_inference_repeated=False)
    (HERE/'label-correction.json').write_text(json.dumps(proof,indent=2),encoding='utf8');print(json.dumps(g,indent=2))
if __name__=='__main__':main()
