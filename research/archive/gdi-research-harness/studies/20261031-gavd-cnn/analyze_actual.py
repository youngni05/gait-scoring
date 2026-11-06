"""Actual frozen scores: independently audited means/AUC, native-FPS and matched results."""
import argparse,collections,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from audit_gavd import HERE,DATA,write_csv,write_json
from prepare_comparison import read_csv,VIEWS
from compare_scores import summarize

def main(args):
    base=HERE/'runs'/args.primary_run/'comparison';scores=read_csv(base/'record_scores.csv');conditions={r['record_id']:r for r in read_csv(HERE/'conditions_shift0.csv')}
    rows=[dict(r,**{k:v for k,v in conditions[r['record_id']].items() if k not in r},score=float(r['score'])) for r in scores]
    # Independent direct CSV arithmetic, without comparison.merge/aggregate.
    raw=read_csv(DATA/'runs'/args.primary_run/'predictions/record_predictions.csv');direct=collections.defaultdict(list)
    for r in raw:direct[r['record_id']].extend([float(r['left_GDI']),float(r['right_GDI'])])
    errors={r['record_id']:abs(r['score']-sum(direct[r['record_id']])/len(direct[r['record_id']])) for r in rows}
    assert all(len(v)==10 for v in direct.values()) and max(errors.values())<1e-10
    strata=[]
    for view in sorted(VIEWS):
        for fps in sorted({float(r['fps_group']) for r in rows if r['view']==view}):
            selected=[r for r in rows if r['view']==view and float(r['fps_group'])==fps]
            result=summarize(selected,5000,20261031)
            if result['status']=='exploratory':
                g=collections.defaultdict(list)
                for r in selected:g[r['label'],r['cluster_id']].append(r['score'])
                n=[sum(v)/len(v) for (label,_),v in g.items() if label=='Normal Gait'];a=[sum(v)/len(v) for (label,_),v in g.items() if label=='Abnormal Gait']
                independent_auc=float(mannwhitneyu(n,a,alternative='two-sided').statistic/(len(n)*len(a)))
                assert abs(independent_auc-result['auc_higher_score_is_normal'])<1e-12
                strata.append(dict(view=view,fps_group=fps,**result))
    pair_scores=[];mapping={r['record_id']:r for r in rows};unavailable=[]
    for p in read_csv(HERE/'matched_pairs_shift0.csv'):
        if p['normal_record'] not in mapping or p['abnormal_record'] not in mapping:unavailable.append(p);continue
        n=mapping[p['normal_record']]['score'];a=mapping[p['abnormal_record']]['score'];pair_scores.append(dict(p,normal_score=n,abnormal_score=a,normal_minus_abnormal=n-a))
    # Repeated sources across views connect pairs; bootstrap these connected groups.
    parent={}
    def find(k):
        parent.setdefault(k,k)
        if parent[k]!=k:parent[k]=find(parent[k])
        return parent[k]
    for r in pair_scores:parent[find(r['normal_cluster'])]=find(r['abnormal_cluster'])
    groups=collections.defaultdict(list)
    for r in pair_scores:groups[find(r['normal_cluster'])].append(r['normal_minus_abnormal'])
    groupdiff=np.array([np.mean(v) for v in groups.values()]);ci=None
    if len(groupdiff)>=2:
        rng=np.random.default_rng(20261031);boot=groupdiff[rng.integers(0,len(groupdiff),size=(5000,len(groupdiff)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
    matched=dict(pairs=len(pair_scores),dependency_groups=len(groupdiff),mean_pair_difference=float(np.mean([r['normal_minus_abnormal'] for r in pair_scores])) if pair_scores else None,equal_dependency_group_difference=float(groupdiff.mean()) if len(groupdiff) else None,dependency_group_ci95=ci,normal_higher_pairs=sum(r['normal_minus_abnormal']>0 for r in pair_scores),model_quality_excluded_pairs=unavailable)
    matched['per_fold_mean_pair_difference']={str(k):float(np.mean([float(mapping[p['normal_record']]['fold_'+str(k)])-float(mapping[p['abnormal_record']]['fold_'+str(k)]) for p in pair_scores])) if pair_scores else None for k in range(5)}
    matched['inference_limit']='Pairs are convenience video samples; a small bootstrap interval cannot establish population/clinical discrimination. No confirmatory significance test.'
    sensitivity=[]
    shifted=read_csv(HERE/'runs'/args.sensitivity_run/'comparison/record_scores.csv')
    for r in shifted:
        if r['record_id'] in mapping:sensitivity.append(dict(record_id=r['record_id'],label=r['label'],primary=mapping[r['record_id']]['score'],shift1=float(r['score']),difference=float(r['score'])-mapping[r['record_id']]['score']))
    by_label={}
    for label in ['Normal Gait','Abnormal Gait']:
        v=[r['score'] for r in rows if r['label']==label];by_label[label]=dict(records=len(v),source_videos=len({r['video_id'] for r in rows if r['label']==label}),dependency_clusters=len({r['cluster_id'] for r in rows if r['label']==label}),record_mean=float(np.mean(v)),record_median=float(np.median(v)),record_min=float(min(v)),record_max=float(max(v)))
    result=dict(status='actual_frozen_model_exploration',by_label=by_label,native_fps_view_strata=strata,strict_condition_matches=matched,frame_shift_sensitivity=dict(common_records=len(sensitivity),mean_absolute_score_change=float(np.mean([abs(r['difference']) for r in sensitivity])),max_absolute_score_change=max(abs(r['difference']) for r in sensitivity)),verification=dict(raw_score_reaggregation_max_error=max(errors.values()),auroc_independently_checked_with_MannWhitneyU=True),actual_GDI_ground_truth=False,causal_or_clinical_claim=False)
    result['primary_run']=args.primary_run;result['sensitivity_run']=args.sensitivity_run
    out=HERE/args.out;out.mkdir(parents=True,exist_ok=True);write_json(out/'actual_results.json',result)
    if pair_scores:write_csv(out/'matched_scores.csv',pair_scores)
    else:
        with (out/'matched_scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            csv.writer(stream).writerow(['normal_record','abnormal_record','normal_score','abnormal_score','normal_minus_abnormal'])
    write_csv(out/'frame_shift_sensitivity.csv',sensitivity);write_csv(out/'scores_with_conditions.csv',rows)
    # Standalone scientific figure, source clusters averaged within view/FPS/label.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True,constrained_layout=True)
    for ax,view in zip(axes.flat,sorted(VIEWS)):
        selected=[r for r in rows if r['view']==view];fpss=sorted({float(r['fps_group']) for r in selected});g=collections.defaultdict(list)
        for r in selected:g[r['label'],float(r['fps_group']),r['cluster_id']].append(r['score'])
        for label,color,marker,offset in [('Normal Gait','#1870a8','o',-.13),('Abnormal Gait','#ce4d31','x',.13)]:
            vals=[(fps,np.mean(v)) for (lab,fps,c),v in g.items() if lab==label]
            ax.scatter([fpss.index(f)+offset for f,s in vals],[s for f,s in vals],c=color,marker=marker,s=42,label=label,alpha=.85)
        ax.axhline(100,ls=':',color='.6',lw=1);ax.set_title(view);ax.set_xticks(range(len(fpss)),[str(f) for f in fpss]);ax.set_xlabel('Decoded native FPS');ax.grid(axis='y',alpha=.18)
    axes[0,0].legend();axes[0,0].set_ylabel('Frozen CNN score');axes[1,0].set_ylabel('Frozen CNN score')
    fig.suptitle('GAVD normal vs abnormal: unchanged 5-fold CNN mean\nDots represent dependency clusters within view/FPS; no true GDI labels',fontsize=13)
    fig.savefig(out/'score_comparison.png',dpi=170);fig.savefig(out/'score_comparison.svg');plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary-run',default='primary');p.add_argument('--sensitivity-run',default='shift1');p.add_argument('--out',default='results');main(p.parse_args())
