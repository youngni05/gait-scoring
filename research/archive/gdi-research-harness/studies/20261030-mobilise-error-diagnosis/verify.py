"""Independent streamed-CSV verification of key diagnostic counts and source identity."""
import csv,json,math,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];BASE=HERE.parent/'20261030-mobilise-bout-regression'
SOURCE=ROOT/'outputs/sppb-tug-candidates-20261030'
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def number(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None
def run():
    result=json.loads((HERE/'diagnosis.json').read_text())
    for path,digest in result['source_sha256'].items():assert sha(path)==digest
    selected={t:set() for t in ['sppb','tug']}
    with (BASE/'features.csv').open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['draw']=='0' and r['band']=='long' and r['k']=='5':
                for target in selected:
                    if r[target+'_ok']=='True':selected[target].add((r['participantid'],r['visit_number']))
    counts={t:dict(all_visits=0,all_tail=0,visits=0,tail=0,tail_people=set(),people=set()) for t in selected}
    components=0;mismatches=0;low_chair_zero=0;low_balance_zero=0;below3=0;above60=0
    with (SOURCE/'All_cohorts_dataset.csv').open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            key=(r['participantid'],r['visit.number'])
            for target,label in [('sppb','total_sppb_score'),('tug','tugtime')]:
                y=number(r[label]);valid=y is not None and ((0<=y<=12 and y.is_integer()) if target=='sppb' else y>0)
                if not valid:continue
                tail=y<=3 if target=='sppb' else y>20;c=counts[target]
                c['all_visits']+=1;c['all_tail']+=tail
                if key not in selected[target]:continue
                c['visits']+=1;c['people'].add(key[0]);c['tail']+=tail
                if tail:c['tail_people'].add(key[0])
                if target=='sppb':
                    values=[number(r[x]) for x in ['balance_score','4mwt_speed_score','5crt_score']]
                    if all(x is not None for x in values):components+=1;mismatches+=sum(values)!=y
                    if tail:low_chair_zero+=values[2]==0;low_balance_zero+=values[0]==0
                else:below3+=y<3;above60+=y>60
    for target,c in counts.items():
        a=result['label_audit'][target]
        assert c['visits']==a['selected_visits']==len(selected[target]) and len(c['people'])==a['selected_people']
        assert c['tail']==a['tail_visits'] and len(c['tail_people'])==a['tail_people']
        assert c['all_visits']==a['all_valid_label_visits'] and c['all_tail']==a['all_valid_tail_visits']
    a=result['label_audit']['sppb']
    assert components==a['component_complete_visits'] and mismatches==a['component_sum_mismatches']==0
    assert low_chair_zero==a['component_groups']['True']['components']['5crt_score']['counts']['0.0']==255
    assert low_balance_zero==a['component_groups']['True']['components']['balance_score']['counts']['0.0']==148
    assert result['label_audit']['tug']['outside_ranges']=={'below3_visits':below3,'above60_visits':above60}
    assert len(result['fold_checks'])==10 and all(x['disjoint'] for x in result['fold_checks'])
    verification={'passed':True,'source_hashes_unchanged':True,'independent_streamed_label_counts_match':True,'sppb_component_sum_checks':components,'sum_mismatches':mismatches,'low_sppb_chair_zero_visits':low_chair_zero,'low_sppb_balance_zero_visits':low_balance_zero,'tug_below3_visits':below3,'tug_above60_visits':above60,'nearest_neighbor_fold_checks':10,'scope':'diagnostic counts and source checks; no new predictive model or causal attribution validated','artifact_sha256':{n:sha(HERE/n) for n in ['diagnosis.json','feature-distributions.csv','cohort-counts.csv']},'script_sha256':{n:sha(HERE/n) for n in ['analyze.py','verify.py']}}
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8');print(json.dumps(verification,indent=2))
if __name__=='__main__':run()
