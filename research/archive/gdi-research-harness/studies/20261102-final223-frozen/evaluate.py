"""Open outcomes once, after all three predictions and the analysis are fixed."""
from common import *
import csv,io,math,datetime
from collections import defaultdict,Counter
NAMES=['CNN_refit','CNN_bag','fusion']

def main():
    protocol=locked();lock=read(HERE/'predictions-locked.json')
    assert sha(BIG/'predictions.npz')==lock['predictions_sha256']
    assert not (HERE/'evaluation-started.json').exists(),'Final reporting already started; use verify.py for technical replay'
    save(HERE/'evaluation-started.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(HERE/'protocol.json'),predictions_sha256=lock['predictions_sha256']))
    member=read(H/'studies/20261027-gdi-linkage/linkage_summary.json')['archive_member']
    arc=module('label_archive',ROOT/'tools/stanford-sample-check/inspect_archive.py')
    payload=arc.read_member(member)
    assert hashlib.sha256(payload).hexdigest()==read(H/'studies/20261027-gdi-linkage/linkage_summary.json')['archive_member_sha256']
    rows=read(HERE/'input-membership.json')['records'];keys={(r['patient_id'],r['exam_id'],s) for r in rows for s in ['L','R']}
    labels=defaultdict(list)
    for r in csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))):
        key=(r['Patient_ID'],r['examid'],r['side'])
        if key in keys:labels[key].append(r['GDI'])
    statuses=[];valid={};label_counts=Counter()
    for r in rows:
        y=[];reason=None
        for s in ['L','R']:
            v=labels[r['patient_id'],r['exam_id'],s]
            if len(v)!=1:reason='missing_label' if not v else 'duplicate_label';break
            try:value=float(v[0])
            except (ValueError,TypeError):reason='nonnumeric_label';break
            if not math.isfinite(value):reason='nonfinite_label';break
            y.append(value)
        if reason is None:valid[r['video_id']]=y
        else:label_counts[reason]+=1
        statuses.append(dict(video_id=r['video_id'],person_id=r['patient_id'],valid=reason is None,reason=reason))
    with np.load(BIG/'predictions.npz') as z:
        original_ids=z['person_id'];original_videos=z['video_id'];keep=np.array([v in valid for v in original_videos])
        ids=original_ids[keep];videos=original_videos[keep];pred={k:z[k][keep] for k in z.files if k not in ['person_id','video_id']}
    y=np.array([valid[v] for v in videos],dtype=np.float64);people=np.unique(ids);assert len(people)>1
    np.savez_compressed(BIG/'scored.npz',y=y,person_id=ids,video_id=videos,**pred)
    errors={k:person_errors(y,pred[k],ids) for k in NAMES}
    cfg=protocol['uncertainty'];rng=np.random.default_rng(cfg['seed']);draws=rng.integers(0,len(people),size=(cfg['draws'],len(people)))
    metrics={}
    for name in NAMES:
        p=pred[name];err=errors[name];mae_samples=err[draws].mean(1)
        moments=np.array([[np.sum(ids==i)*2,y[ids==i].sum(),p[ids==i].sum(),(y[ids==i]**2).sum(),(p[ids==i]**2).sum(),(y[ids==i]*p[ids==i]).sum()] for i in people])
        ss=moments[draws].sum(1);n,sy,sp,syy,spp,syp=ss.T
        rr=(syp-sy*sp/n)/np.sqrt((syy-sy*sy/n)*(spp-sp*sp/n))
        foldkeys=['CNN_%d_refit'%f if name=='CNN_refit' else 'CNN_%d_bag'%f if name=='CNN_bag' else 'fusion_%d'%f for f in range(5)]
        foldmae=[float(person_errors(y,pred[k],ids).mean()) for k in foldkeys]
        metrics[name]=dict(person_equal_mae=float(err.mean()),mae95=np.percentile(mae_samples,[2.5,97.5]).tolist(),pearson=float(np.corrcoef(y.ravel(),p.ravel())[0,1]),pearson95=np.percentile(rr[np.isfinite(rr)],[2.5,97.5]).tolist(),person_equal_bias=float(np.mean([(p[ids==i]-y[ids==i]).mean() for i in people])),per_origin_fold_mae=foldmae,per_origin_fold_mae_sample_sd=float(np.std(foldmae,ddof=1)))
    comparisons={}
    for ref in ['CNN_bag','CNN_refit']:
        delta=errors['fusion']-errors[ref];ci=np.percentile(delta[draws].mean(1),[2.5,97.5]);mean=float(delta.mean())
        verdict='fusion_better' if mean<0 and ci[1]<0 else 'CNN_reference_better' if ci[0]>0 else 'inconclusive'
        comparisons['fusion_minus_'+ref]=dict(mae_delta=mean,relative_mae_reduction_percent=-100*mean/errors[ref].mean(),paired_person_bootstrap95=ci.tolist(),verdict=verdict,people_improved=int((delta<-1e-12).sum()),people_worsened=int((delta>1e-12).sum()),people_tied=int((np.abs(delta)<=1e-12).sum()))
    truth=np.array([y[ids==i].mean() for i in people]);bands=[]
    for low,high,label in [(-np.inf,60,'below60'),(60,70,'60to70'),(70,80,'70to80'),(80,90,'80to90'),(90,np.inf,'90plus')]:
        ix=(truth>=low)&(truth<high)
        bands.append(dict(band=label,people=int(ix.sum()),mae={k:float(v[ix].mean()) if ix.any() else None for k,v in errors.items()}))
    report=dict(protocol_sha256=sha(HERE/'protocol.json'),models_sha256=sha(HERE/'models.json'),predictions_sha256=lock['predictions_sha256'],scored_sha256=sha(BIG/'scored.npz'),reservation_people=223,reservation_records=378,input_eligible_people=lock['people'],input_eligible_records=lock['records'],evaluated_people=len(people),evaluated_records=len(ids),evaluated_record_sides=len(ids)*2,label_exclusions=dict(label_counts),input_eligible_records_excluded_for_labels=int((~keep).sum()),metrics=metrics,comparisons=comparisons,person_mean_truth_bands=bands,bootstrap=cfg,numeric_reserved_labels_opened=True,optimizer_steps=0,post_evaluation_tuning=False,notes=['Primary predictions average five frozen fold predictors; per-fold SD is a diagnostic, not five independent holdout repetitions.','Same-source unseen-person evaluation, not older-adult or external-domain validation.','Confidence intervals resample fixed predictions, not uncertainty from retraining or all prior model selection.'])
    save(HERE/'label-audit.json',dict(statuses=statuses,label_exclusions=dict(label_counts),annotation_sha256=hashlib.sha256(payload).hexdigest()))
    save(HERE/'results.json',report)
    fields=['person_id','video_id','true_left','true_right']+[k+s for k in NAMES for s in ['_left','_right']]
    with (HERE/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for j in range(len(ids)):
            row=dict(person_id=ids[j],video_id=videos[j],true_left=y[j,0],true_right=y[j,1])
            for k in NAMES:row.update({k+'_left':pred[k][j,0],k+'_right':pred[k][j,1]})
            w.writerow(row)
    lines=['# 예약 223명 고정 모델 최종 평가','',f"평가 인원 {len(people)}명, 기록 {len(ids)}개. 세 모델은 같은 입력·정답에서 비교했다.",'','| 구성 | 사람 동일 비중 MAE | MAE 95% 범위 | 상관계수 |','|---|---:|---|---:|']
    for k in NAMES:
        m=metrics[k];lines.append(f"| {k} | {m['person_equal_mae']:.6f} | {m['mae95'][0]:.6f}~{m['mae95'][1]:.6f} | {m['pearson']:.6f} |")
    c=comparisons['fusion_minus_CNN_bag'];lines+=['',f"주 비교(결합−CNN 평균): {c['mae_delta']:.6f}점. 사람 단위 95% 범위 {c['paired_person_bootstrap95'][0]:.6f}~{c['paired_person_bootstrap95'][1]:.6f}점. 판정: {c['verdict']}.",'','모델·보정식·결합 비율과 제외 기준은 결과 확인 전에 고정했다. 예측 파일을 잠근 뒤 실제 GDI 정답을 열었다. 결과를 이용한 재학습·설정 변경은 하지 않았다. 고령자·외부 촬영 환경의 검증과는 구분한다.','', '상세 조건은 protocol.json, 입력 제외는 input-audit.json, 수치 결과는 results.json, 독립 검산은 verification.json에 남긴다.']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['evaluated_people','evaluated_records','metrics','comparisons']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
