"""Unbiased outer-test curves; report draws separately before averaging their errors."""
import os,sys,csv,json,hashlib
import numpy as np
from common import load,save,sha,data,mask,check_plan,metrics
import manage

def main():
    out=manage.HERE/'analysis';cnn=load(out/'round0/verification.json');rf=load(out/'rf/verification.json')
    assert cnn['passed'] and cnn['verified_fits']==1800 and cnn['final_models']==450
    assert rf['passed'] and rf['saved_replayed_fits']==1050 and rf['final_models']==150
    assert cnn['plan_sha256']==sha(manage.BUNDLE/'plan.json')
    assert rf['plan_sha256']==sha(manage.HERE/'plan.json')
    assert load(manage.BUNDLE/'plan.json')==load(manage.HERE/'plan.json'), 'Decoded participant plans differ'
    for f in rf['files']:assert sha(f['path'])==f['sha256']
    d=data(manage.BUNDLE/'target.npz');plan=load(manage.HERE/'plan.json');check_plan(plan,d)
    rows=[]
    for folder in ('round0','rf'):
        for r in csv.DictReader((out/folder/'person-predictions.csv').open(encoding='utf8')):
            r.update(y=float(r['y']),prediction=float(r['prediction']),seed=int(r['seed']),draw=int(r['draw']));rows.append(r)
    assert len(rows)==5280
    ids=sorted(np.unique(d['ids']).tolist());truth=np.array([d['y'][d['ids']==p].mean() for p in ids])
    for case in plan['cases']:
        yy=np.array([d['y'][d['ids']==p].mean() for p in case['people']])
        for name,value in [('Mean',float(yy.astype(np.float64).mean())),('Median',float(np.median(yy)))]:
            rows += [dict(id=p,y=float(d['y'][d['ids']==p].mean()),prediction=value,fold=case['fold'],case=case['key'],size=str(case['size']),draw=case['draw'],model=name,seed=0) for p in case['test']]
    index={}
    for r in rows:
        key=(r['model'],str(r['size']),r['draw'],r['seed'],r['id']);assert key not in index;index[key]=r
    summaries={};error_vectors={};predictions={};curve_rows=[]
    models=['RF','CNN32','CNN16','CNN8','Mean','Median']
    for model in models:
        summaries[model]={}
        for size in ('8','16','24','full'):
            draws=range(3) if size!='full' else [-1];draw_results=[];errors=[];seedstd=[]
            for draw in draws:
                seeds=(20261101,20261102,20261103) if model not in ('Mean','Median') else (0,)
                ps=np.array([[index[(model,size,draw,seed,p)]['prediction'] for p in ids] for seed in seeds])
                np.testing.assert_array_equal(truth,[index[(model,size,draw,seeds[0],p)]['y'] for p in ids])
                mean=ps.mean(0);person=[dict(id=p,y=float(y),prediction=float(v)) for p,y,v in zip(ids,truth,mean)]
                m=metrics(person);draw_results.append(dict(draw=draw,metrics=m,seed_metrics={str(seed):metrics([dict(y=float(y),prediction=float(v)) for y,v in zip(truth,pred)]) for seed,pred in zip(seeds,ps)}))
                errors.append(np.abs(mean-truth));seedstd.append(float(ps.std(0).mean()));predictions[(model,size,draw)]=mean
            mean_metrics={k:float(np.mean([r['metrics'][k] for r in draw_results])) for k in draw_results[0]['metrics']}
            summaries[model][size]=dict(mean_metrics=mean_metrics,draw_MAE_std=float(np.std([r['metrics']['MAE'] for r in draw_results])),mean_person_seed_std=float(np.mean(seedstd)),draws=draw_results)
            error_vectors[(model,size)]=np.mean(errors,axis=0)
            curve_rows.append(dict(model=model,size=size,**mean_metrics,draw_MAE_std=summaries[model][size]['draw_MAE_std'],mean_person_seed_std=float(np.mean(seedstd))))
    reproduction={}
    for model,path,key in [('CNN32','reference-cnn.csv','scratch'),('RF','reference-rf.csv','RF')]:
        reference={r['id']:float(r[key]) for r in csv.DictReader((manage.HERE/path).open(encoding='utf8'))}
        old=np.array([reference[p] for p in ids]);new=predictions[(model,'full',-1)]
        np.testing.assert_allclose(new,old,atol=1e-4,rtol=0,err_msg='Full-condition reproduction mismatch '+model)
        reproduction[model]=dict(passed=True,max_absolute_difference=float(np.max(np.abs(new-old))))
    rng=np.random.RandomState(20261101);boot=rng.randint(0,44,(10000,44));paired={}
    def interval(diff):return dict(MAE_difference=float(diff.mean()),conditional_person_bootstrap95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
    for model in models:
        for size in ('8','16','24','full'):
            paired[model+'_minus_RF_n'+size]=interval(error_vectors[(model,size)]-error_vectors[('RF',size)])
        for small in ('8','16','24'):
            paired[model+'_full_minus_n'+small]=interval(error_vectors[(model,'full')]-error_vectors[(model,small)])
    result=dict(passed=True,people=44,CNN_plan_sha256=cnn['plan_sha256'],RF_plan_sha256=rf['plan_sha256'],logical_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),metrics=summaries,paired=paired,full_reproduction=reproduction,CNN_verification_sha256=sha(out/'round0/verification.json'),RF_verification_sha256=sha(out/'rf/verification.json'),scope='Outer people held fixed across sizes/models. Three seeds averaged within each draw; report mean errors across draws, not a nine-model ensemble. Conditional intervals ignore model fitting/draw uncertainty;44person repeatedly examined development data.')
    save(out/'comparison.json',result)
    with (out/'curves.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(curve_rows[0]));w.writeheader();w.writerows(curve_rows)
    lines=['# 사람 수별 성능 곡선과 작은 CNN 비교','',
           '44명·220회 자료에서 같은 평가 사람을 유지하고 최종 학습 인원을8·16·24명·전체35–36명으로 늘렸다. 각 조건 안의 내부3분할로만 epoch와 RF 설정을 골랐다. 작은 집단은 포함 관계를 갖는3추출이며, 추출별3seed평균예측의 오차를 구한 뒤 추출 간 평균했다. 새 사람을 확보한 결과가 아니다.','',
           '| 방법 | 8명 MAE | 16명 MAE | 24명 MAE | 전체 MAE | 전체 ±1점 | 전체5–8점 MAE |','|---|---:|---:|---:|---:|---:|---:|']
    for model in models:
        values=[summaries[model][size]['mean_metrics']['MAE'] for size in ('8','16','24','full')];full=summaries[model]['full']['mean_metrics']
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.3f |'%(model,*values,full['within1']*100,full['low5_8_MAE']))
    lines+=['','CNN32는 기존 구조이며CNN16·CNN8은 채널 수만 줄였다. RF는 같은 좌표의 시간 요약36개를 쓴다. BN·정답 정규화·RF scaler는 실제 학습 사람으로만 계산했다. 전체 조건CNN32와 RF는 이전 검산 예측과1e-4 이내로 재현되는지 확인했다.','',
            '기본 학습 곡선의 오차 막대는 사람 추출3회 사이의 표준편차이다. 전체 조건은 같은 사람 전체 집단이라 추출을 중복하지 않았다. 막대를 전체 불확실성의 신뢰구간으로 해석하지 않는다. 초기값·추출별 지표,점수대 오차와 고정 예측의 조건부 사람 재표집 구간은 comparison.json에 있다.','',
            '44명의 점수는5–12점이고34명이9–12점이다. 같은 사람의5회 보행은 한꺼번에 분리했다. 기능점수 정보가 없는 새 사전학습 사람을 늘린 실험과 구분한다. 수백 명에서의 개선 크기나 실제 RGB·임상 성능을 확정할 수 없다. 결과를 보고 추가 GPU 탐색을 자동으로 시작하지 않는다.']
    lines+=['','이번 범위에서는 사람 수 보완을 우선할 근거가 강해졌다. 학습8명에서 전체35–36명으로 늘렸을 때 RF의 평균 MAE는%.3f→%.3f점(%.1f%% 감소),CNN32는%.3f→%.3f점(%.1f%% 감소)이었다.'%(summaries['RF']['8']['mean_metrics']['MAE'],summaries['RF']['full']['mean_metrics']['MAE'],100*(1-summaries['RF']['full']['mean_metrics']['MAE']/summaries['RF']['8']['mean_metrics']['MAE']),summaries['CNN32']['8']['mean_metrics']['MAE'],summaries['CNN32']['full']['mean_metrics']['MAE'],100*(1-summaries['CNN32']['full']['mean_metrics']['MAE']/summaries['CNN32']['8']['mean_metrics']['MAE'])), '',
            '작은 CNN16은8·16명 조건에서 기존CNN32보다 평균 오차가 작았지만, 전체 조건에서는CNN32 %.3f점보다CNN16 %.3f점·CNN8 %.3f점의 오차가 컸다. 채널 수만 줄이는 방법으로 전체 성능을 개선하지는 못했다. 학습률·드롭아웃은 동일하게 유지했으므로 소형 모델을 별도로 최적화했을 때의 최선 성능까지 부정하는 결과는 아니다.'%(summaries['CNN32']['full']['mean_metrics']['MAE'],summaries['CNN16']['full']['mean_metrics']['MAE'],summaries['CNN8']['full']['mean_metrics']['MAE']), '',
            '전체 조건의 초기값별 사람 예측 표준편차 평균은RF %.3f점,CNN32 %.3f점,CNN16 %.3f점,CNN8 %.3f점이었다. CNN을 줄여 초기값 민감도가 낮아진다는 기대도 이번 비교에서는 확인되지 않았다.'%(summaries['RF']['full']['mean_person_seed_std'],summaries['CNN32']['full']['mean_person_seed_std'],summaries['CNN16']['full']['mean_person_seed_std'],summaries['CNN8']['full']['mean_person_seed_std']), '',
            'RF도 어떤 사람을 뽑았는지에 영향을 받았다.8명 조건의 추출별 MAE는%.3f~%.3f점이었다. 초기값에 안정적인 것과 사람 표본에 안정적인 것은 구분해야 한다. 전체 조건의 추출 표준편차0은 같은 전체 집단을 중복 학습하지 않았다는 뜻이며 불확실성이 없다는 뜻은 아니다.'%(min(q['metrics']['MAE'] for q in summaries['RF']['8']['draws']),max(q['metrics']['MAE'] for q in summaries['RF']['8']['draws'])), '',
            'RF와CNN32는 계속 비교 기준으로 유지할 가치가 있다. 전체 MAE 차이는%.3f점이고 고정 예측의 사람 재표집95%% 구간은%.3f~%.3f점으로 우열을 확정하기 어렵다.5–8점10명의 MAE는RF %.3f점,CNN32 %.3f점이었다.'%(paired['CNN32_minus_RF_nfull']['MAE_difference'],*paired['CNN32_minus_RF_nfull']['conditional_person_bootstrap95'],summaries['RF']['full']['mean_metrics']['low5_8_MAE'],summaries['CNN32']['full']['mean_metrics']['low5_8_MAE']), '',
            '후속 판단은 실제 기능점수가 연결된 새로운 사람을 늘리고 부족한 점수대를 보완하는 쪽에 무게를 둔다. 이번 실험은 점수대 비율을 가능한 한 유지하며 인원 수를 비교했으므로, 낮은 점수 비율을 늘리는 효과 자체를 검증한 실험은 아니다. 인원 증가 효과에는 해당 집단 안에서 epoch·RF 설정을 선택하는 과정의 변화도 포함한다. 추가 GPU 학습이나 신규 자료 수집은 이번 완료 절차에서 실행하지 않았다.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    os.environ['MPLCONFIGDIR']=str(manage.DATA/'cache/matplotlib')
    sys.path.append(str(manage.WORKSPACE/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');x=[8,16,24,float(np.mean([c['final_training_people'] for c in plan['cases'] if c['size']=='full']))]
    for model in models:
        axes[0].errorbar(x,[summaries[model][n]['mean_metrics']['MAE'] for n in ('8','16','24','full')],yerr=[summaries[model][n]['draw_MAE_std'] for n in ('8','16','24','full')],marker='o',capsize=3,label=model)
        if model not in ('Mean','Median'):axes[1].plot(x,[summaries[model][n]['mean_person_seed_std'] for n in ('8','16','24','full')],marker='o',label=model)
    for ax in axes:ax.set_xticks(x,['8','16','24','35-36']);ax.set_xlabel('Final training people');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Outer-test person MAE (SPPB points)');axes[1].set_ylabel('Mean person prediction SD across seeds')
    fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
    print(dict(passed=True,full_reproduction=reproduction,report=str(out/'RESULTS.md')))
if __name__=='__main__':main()
