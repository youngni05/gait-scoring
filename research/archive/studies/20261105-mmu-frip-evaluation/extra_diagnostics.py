"""Read-only label and empirical individual-error diagnostics; no refitting."""
import csv,json
import numpy as np
import prepare as p
H=p.H;B=p.B

def main():
    d=np.load(B/'input.npz');pred=np.load(B/'verified-predictions.npz');receipt=json.loads((H/'fit-receipt.json').read_text())
    rows=list(csv.DictReader((p.S/'Extracted/final_gait_feature.csv').open(encoding='utf-8-sig')));meta={r['ID']:r for r in rows}
    age_discrepancies=[]
    for pid in d['ids']:
        row=meta[pid];age=float(row['age']);expected=sum(age>=b for b in [60,70,80]);provided=float(row['ageGroup'])
        if expected!=provided:age_discrepancies.append({'id':str(pid),'age':age,'provided_ageGroup':provided,'points_under_standard_age_rule':expected})
    diag=json.loads((H/'diagnosis.json').read_text());flagged=set(a['id'] for a in age_discrepancies)|set(a['id'] for a in diag['label_composition']['mismatches'])
    mask=np.array([pid not in flagged for pid in d['ids']]);sensitivity={a:float(abs(pred[a][mask]-d['y'][mask]).mean()) for a in ['A','B','C','D']}
    intervals={}
    for j,arm in enumerate(['A','B','C','R']):
        radius=np.empty(125)
        for fold,row in enumerate(receipt['folds']):
            ctx=np.load(row['context']);tr=ctx['train'] if 'train' in ctx else np.array(row['train']);te=np.array(row['test'])
            residual=abs(ctx['innerOOF'][:,j]-d['y'][tr]);q=float(np.quantile(residual,.9,method='higher'));radius[te]=q
        intervals[arm]={g:{'people':int(sum(m)),'nominal':.9,'empirical_coverage':float(np.mean(abs(pred[arm][m]-d['y'][m])<=radius[m])),'median_full_width_points':float(np.median(radius[m]*2)),'median_radius_points':float(np.median(radius[m]))} for g,m in [('all',np.ones(125,bool)),('age65plus',d['age']>=65)]}
    batches={}
    for name,mask_batch in [('original_ID001_to076',np.array([int(pid[1:])<=76 for pid in d['ids']])),('expanded_ID077plus',np.array([int(pid[1:])>76 for pid in d['ids']]))]:
        batches[name]={'people':int(sum(mask_batch)),'mean_score':float(d['y'][mask_batch].mean()),'score14plus':int(sum(d['y'][mask_batch]>=14)),'CNN_MAE':float(abs(pred['C'][mask_batch]-d['y'][mask_batch]).mean())}
    out={'status':'descriptive_diagnostics_no_primary_change','standard_age_scoring_source':'https://www.rwjbh.org/documents/nursing/nursing-resources/Johns-Hopkins-Fall-Risk-Assessment-Tool.pdf',
        'age_point_mismatches':age_discrepancies,'interpretation':'ProvidedageGroupdoesnotalwaysmatchrecordedageunderstandardtool. Unknownwhetheradaptedscoringormetadataerror. Donotautomaticallycorrect.',
        'binary_label_matches_score_threshold6':all(int(float(r['fall_risk']))==int(float(r['score'])>=6) for r in rows),
        'same_frozen_predictions_excluding_all_label_flagged_rows':{'excluded':sorted(flagged),'remaining':int(sum(mask)),'MAE':sensitivity,'not_retrained_or_primary':True},
        'release_cohort_diagnostic':batches,'cohort_limit':'Allfiveupper-tailpeopleareinexpandedbatch. Descriptiveonly,noproofbatchcausestheerrors;notleave-batch-outvalidation.',
        'individual_error_intervals':intervals,'interval_limits':'EmpiricalinnerOOFresidual90percentinterval,notformalcoverageguaranteeorclinicalchangeMDC. Notcalibratedonoutertest. Cross-fitmodelsdiffersfromfinalfit. Negativeendpointsnottruncatedforwidth.'}
    p.save(H/'extra-diagnostics.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
