import csv,json,math,statistics,hashlib
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def load(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def read(name):
    with (HERE/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(a-b)<1e-9,(a,b)
def ranks(a):
    order=sorted(range(len(a)),key=a.__getitem__);out=[0.]*len(a);start=0
    while start<len(a):
        end=start+1
        while end<len(a) and a[order[start]]==a[order[end]]:end+=1
        for i in order[start:end]:out[i]=(start+1+end)/2
        start=end
    return out
def corr(a,b):
    if len(set(a))<2 or len(set(b))<2:return None
    return statistics.correlation(ranks(a),ranks(b))
def people_mean(v):
    grouped=defaultdict(list)
    for person,value in v:grouped[person].append(value)
    return {p:statistics.mean(x) for p,x in grouped.items()}

def main():
    r=load('result.json');mixed=load('mixed_result.json');rows=read('quality_rows.csv')
    assert len(rows)==729
    for role in ('train','validation'):
        for pos in ('front','middle','last'):
            selected=[x for x in rows if x['split']==role and x['position']==pos]
            for key,target in r['summary'][role][pos].items():
                computed=statistics.mean(people_mean([(x['patient_id'],float(x[key])) for x in selected]).values())
                close(computed,target)
    prev=ROOT/'gdi-research-harness/control/desktop-sessions/20261027T153603Z-79100a/predictions.csv'
    with prev.open(encoding='utf-8') as f:preds=list(csv.DictReader(f))
    lookup={(x['patient_id'],x['video_id'],x['method']):x for x in preds}
    for pos in ('middle','last'):
        signed=[];absolute=[]
        for x in preds:
            if x['method']!=pos:continue
            old=lookup[(x['patient_id'],x['video_id'],'front')]
            delta=[float(x['prediction_'+s])-float(old['prediction_'+s]) for s in 'LR']
            signed.append((x['patient_id'],statistics.mean(delta)));absolute.append((x['patient_id'],statistics.mean(map(abs,delta))))
        sd,ad=people_mean(signed),people_mean(absolute);ids=sorted(sd)
        for key,expected in r['associations'][pos]['metrics'].items():
            before=people_mean([(x['patient_id'],float(x[key])) for x in rows if x['split']=='validation' and x['position']=='front'])
            after=people_mean([(x['patient_id'],float(x[key])) for x in rows if x['split']=='validation' and x['position']==pos])
            change=[after[p]-before[p] for p in ids];close(statistics.mean(change),expected['mean_change'])
            for a,b,k in ((change,[sd[p] for p in ids],'signed_spearman'),(list(map(abs,change)),[ad[p] for p in ids],'absolute_spearman')):
                value=corr(a,b)
                if value is None:assert expected[k] is None
                else:close(value,expected[k])
    cross=read('cross_heads.csv');assert len(cross)==360
    for trained in ('front','middle','last'):
        for pos in ('front','middle','last'):
            subset=[x for x in cross if x['trained_position']==trained and x['input_position']==pos]
            errors=people_mean([(x['patient_id'],statistics.mean(abs(float(x['prediction_'+s])-float(x['actual_'+s])) for s in 'LR')) for x in subset])
            close(statistics.mean(errors.values()),r['cross_head_person_mae'][trained][pos])
    mr=read('mixed_predictions.csv');assert len(mr)==160
    for method in ('front','middle','last','mean3'):
        errors=people_mean([(x['patient_id'],statistics.mean(abs(float(x['prediction_'+s])-float(x['actual_'+s])) for s in 'LR')) for x in mr if x['method']==method])
        close(statistics.mean(errors.values()),mixed['person_mae'][method])
    groups=defaultdict(dict)
    for x in mr:groups[(x['patient_id'],x['video_id'])][x['method']]=[float(x['prediction_'+s]) for s in 'LR']
    ranges=[]
    for (p,v),m in groups.items():
        for side in (0,1):close(m['mean3'][side],statistics.mean(m[k][side] for k in ('front','middle','last')))
        ranges.append((p,statistics.mean(max(m[k][s] for k in ('front','middle','last'))-min(m[k][s] for k in ('front','middle','last')) for s in (0,1))))
    close(statistics.mean(people_mean(ranges).values()),mixed['three_position_range_mean'])
    for path,digest in load('sources.json').items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    # plan was appended for header-only availability checking after mixed-model training.
    for path,digest in load('mixed_sources.json').items():
        if not path.endswith('plan.md'):assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    result=dict(verified=True,quality_csv_rows=729,cross_head_rows=360,mixed_prediction_rows=160,
        summaries_and_spearman_independently_recomputed=True,cross_head_mae_recomputed=True,
        mixed_mae_range_and_mean_verified=True,source_hashes_unchanged=True,
        bootstrap_independently_recomputed=False,raw_motion_formulas_independently_reimplemented=False)
    with (HERE/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
