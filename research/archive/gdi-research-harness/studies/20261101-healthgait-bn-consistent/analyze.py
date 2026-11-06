"""After complete independent replay, compare all person-level OOF predictions."""
import csv
import os
import sys
import json
import numpy as np
import manage


def main():
    root=manage.HERE/'analysis'
    cnn=manage.load(root/'round0/verification.json')
    baseline=manage.load(root/'baselines/verification.json')
    assert cnn['passed'] and cnn['bn_verified_fits']==cnn['all_fits']==180
    assert baseline['passed'] and baseline['saved_and_replayed_fits']==140
    rows=list(csv.DictReader((root/'round0/person-predictions.csv').open(encoding='utf8')))
    br={r['id']:r for r in csv.DictReader((root/'baselines/person-predictions.csv').open(encoding='utf8'))}
    ids=[r['id'] for r in rows];truth=np.array([float(r['SPPB']) for r in rows])
    np.testing.assert_array_equal(truth,[float(br[p]['SPPB']) for p in ids])
    values={a:np.array([float(r[a]) for r in rows]) for a in ('scratch','gdi','health')}
    values.update({a:np.array([float(br[p][a]) for p in ids]) for a in ('Ridge','RF','mean','median')})
    names={'scratch':'처음부터 CNN','gdi':'GDI → CNN','health':'Health&Gait → CNN','Ridge':'선형 회귀(Ridge)','RF':'랜덤 포레스트','mean':'학습 사람 평균','median':'학습 사람 중앙값'}
    rng=np.random.RandomState(20261101);boot=rng.randint(0,len(ids),(10000,len(ids)))
    metrics={};comparisons={}
    bands=[('5–8',truth<=8),('9–10',(truth>=9)&(truth<=10)),('11–12',truth>=11)]
    for arm,pred in values.items():
        err=pred-truth
        metrics[arm]=dict(MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),
                          bands={label:dict(people=int(mask.sum()),MAE=float(np.abs(err[mask]).mean())) for label,mask in bands})
        for reference in ('median','RF'):
            diff=np.abs(err)-np.abs(values[reference]-truth)
            comparisons[arm+'_minus_'+reference]=dict(MAE_difference=float(diff.mean()),conditional_person_bootstrap95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
    previous=manage.HERE.parent/'20261101-healthgait-transfer/analysis/round0/verification.json'
    old=manage.load(previous)
    changes={a:dict(previous_MAE=old['metrics'][a]['ensemble_MAE'],corrected_MAE=metrics[a]['MAE'],difference=metrics[a]['MAE']-old['metrics'][a]['ensemble_MAE']) for a in ('scratch','gdi','health')}
    selected=manage.load(root/'round0/selected-configs.json')
    epochs={a:dict(median=float(np.median([r['epochs'] for r in selected if r['arm']==a])),range=[min(r['epochs'] for r in selected if r['arm']==a),max(r['epochs'] for r in selected if r['arm']==a)]) for a in ('scratch','gdi','health')}
    curve_root=manage.DATA/'received/round0/originals/runs/round0'
    learning_curves={}
    for arm in ('scratch','gdi','health'):
        curves=[]
        for path in curve_root.glob('fold*/'+arm+'-s*/candidate0/inner*/curve.json'):
            c=manage.load(path);best=min(c,key=lambda v:(v['validation_MAE'],v['epoch']))
            identity=manage.load(path.parent/'identity.json')
            curves.append(dict(first=c[0]['validation_MAE'],best=best['validation_MAE'],last=c[-1]['validation_MAE'],best_epoch=best['epoch'],epochs=len(c),train_people=len(identity['train_people']),trainloss_best=best['train_loss'],trainloss_last=c[-1]['train_loss']))
        assert len(curves)==45
        learning_curves[arm]=dict(inner_fits=45,train_people_range=[min(c['train_people'] for c in curves),max(c['train_people'] for c in curves)],median_best_epoch=float(np.median([c['best_epoch'] for c in curves])),best_at_epoch1=sum(c['best_epoch']==1 for c in curves),max_epoch_reached=sum(c['epochs']==80 for c in curves),mean_validation_MAE_first=float(np.mean([c['first'] for c in curves])),mean_validation_MAE_best=float(np.mean([c['best'] for c in curves])),mean_validation_MAE_last=float(np.mean([c['last'] for c in curves])),trainloss_falls_but_validation_worsens_after_best=sum(c['trainloss_last']<c['trainloss_best'] and c['last']>c['best'] for c in curves))
    result=dict(passed=True,people=44,metrics=metrics,paired_comparisons=comparisons,previous_round0_comparison=changes,selected_epochs=epochs,learning_curves=learning_curves,
                verification_sha256=manage.sha(root/'round0/verification.json'),baseline_verification_sha256=manage.sha(root/'baselines/verification.json'),
                scope='Exploratory repeatedly examined44-person CV; conditional bootstrap ignores training variability. Health source also retrained under corrected BN. No independent clinical or RGB test.')
    manage.save(root/'comparison.json',result)
    lines=['# BN 일관 처리 재비교','', '44명·220회 자료의 같은 사람별5분할을 비교했다. CNN과 RF는 초기값3개의 평균 예측이며, Ridge는 결정적 모델이다. 선형 회귀·RF는 같은 좌표의 시간 요약36개를 사용한다.','',
           '| 방법 | MAE(점) | RMSE(점) | 5–8점 MAE |','|---|---:|---:|---:|']
    for arm in values:
        m=metrics[arm];lines.append('| %s | %.3f | %.3f | %.3f |'%(names[arm],m['MAE'],m['RMSE'],m['bands']['5–8']['MAE']))
    lines+=['','BN 통계는 학습 사람만으로 다시 계산했고, 내부 검증과 최종 예측에 동일한 절차를 적용했다. 저장된180개 CNN 모델의 통계·예측과140개 기준선 학습의 저장 재추론을 검산했다.','',
            '이전 초기 비교와의 차이는 아래와 같다. Health&Gait는 사전학습부터 같은 BN 절차로 다시 학습했으므로, 이 차이를 대상 자료의 BN 수정 효과 하나로만 해석할 수 없다.','',
            '| CNN 초기화 | 이전 MAE | 수정 후 MAE | 차이(수정−이전) |','|---|---:|---:|---:|']
    for arm,c in changes.items():lines.append('| %s | %.3f | %.3f | %+.3f |'%(names[arm],c['previous_MAE'],c['corrected_MAE'],c['difference']))
    lines+=['','점수별 오차·초기값별 성능·선택 epoch와 사람별 오차 차이는 comparison.json 및 각 verification.json에 기록했다. 사람 단위 재표집 구간은 고정된 예측에 대한 탐색적 구간이며 학습 변동 전체를 포함하지 않는다.','',
            '기능점수 정답은 여전히44명이며34명이9–12점이다. 관절 입력은 Vicon 표식을 투영한 좌표이다. 반복해서 살펴본 개발 자료의 결과이므로 일반인 영상·현장·임상 성능으로 확대할 수 없다.']
    lines+=['','이번 관측에서는 BN 수정 뒤 세 CNN 모두 중앙값보다 평균 오차가 작았다. 특히 GDI 초기화의 큰 오차가 줄어, 이전 성능 저하를 사람 수나 점수 분포만으로 설명할 수 없음을 확인했다. 다만 Health&Gait 사전학습은 처음부터 학습하는 조건을 넘지 못했다.','',
            'RF와 처음부터 학습한 CNN의 MAE 차이는 %.3f점이며, 고정 예측의 사람 재표집95%% 구간은 %.3f~%.3f점이다. RF의 일반적 우월성이 확정된 것으로 해석하지 않는다.'%(comparisons['scratch_minus_RF']['MAE_difference'],*comparisons['scratch_minus_RF']['conditional_person_bootstrap95']), '',
            '낮은5–8점10명의 오차는 처음부터 CNN %.3f점, Health&Gait CNN %.3f점이다. 처음부터 CNN도 전체 예측 범위가 %.2f~%.2f점으로 좁아 높은 점수를 낮게 추정하는 문제가 남았다.'%(metrics['scratch']['bands']['5–8']['MAE'],metrics['health']['bands']['5–8']['MAE'],values['scratch'].min(),values['scratch'].max()), '',
            '초기값을 바꿀 때 사람별 예측의 표준편차 평균은 CNN %.3f~%.3f점, RF %.3f점이다. 표본이 적은 조건에서 CNN의 결과가 덜 안정적이었다.'%(min(cnn['metrics'][a]['seed_prediction_std_mean'] for a in ('scratch','gdi','health')),max(cnn['metrics'][a]['seed_prediction_std_mean'] for a in ('scratch','gdi','health')),baseline['metrics']['RF']['seed_prediction_std_mean']), '',
            '학습 곡선에서 학습 손실은 줄지만 검증 오차가 다시 커진 경우가 각45개 내부 학습 중 처음부터%d개, GDI%d개, Health&Gait%d개였다. 내부 학습 집단은23~24명이다. 단순히 더 오래 학습하는 것보다 모델 복잡도와 일반화 문제를 살펴볼 근거다. 최선 epoch를 검증에서 선택한 구조 때문에 생기는 선택 효과도 있어, 이 개수만으로 과적합의 원인을 확정하지 않는다.'%(learning_curves['scratch']['trainloss_falls_but_validation_worsens_after_best'],learning_curves['gdi']['trainloss_falls_but_validation_worsens_after_best'],learning_curves['health']['trainloss_falls_but_validation_worsens_after_best']), '',
            '후속 제안은 RF와 수정된 처음부터 CNN을 기준으로 유지하고, 더 작은 CNN과 사람 단위 안정성 검증을 비교하는 것이다. 실제 기능점수가 연결된 사람 수를 늘리는 방향도 필요하다. 이번 승인된 BN 재비교는 여기까지이며 새 GPU 탐색은 제출하지 않았다.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    os.environ['MPLCONFIGDIR']=str(manage.DATA/'cache/matplotlib')
    sys.path.append(str(manage.WORKSPACE/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    keys=['scratch','gdi','health','Ridge','RF']
    ax=axes.flat[0]
    ax.barh(keys+["median"],[metrics[a]['MAE'] for a in keys+['median']],color=['#5289c7']*3+['#669b70']*2+['#a0a0a0'])
    ax.set_xlabel('Person MAE (SPPB points)');ax.invert_yaxis();ax.set_title('Same 44 people, five outer folds')
    for ax,arm in zip(list(axes.flat)[1:],keys):
        ax.scatter(truth,values[arm],alpha=.65,s=25)
        ax.plot([4,13],[4,13],color='gray',linestyle='--',linewidth=1)
        ax.set(xlim=(4,13),ylim=(min(4,float(values[arm].min())-.5),max(13,float(values[arm].max())+.5)),xlabel='Actual SPPB',ylabel='Predicted SPPB',title='%s | MAE %.3f'%(arm,metrics[arm]['MAE']))
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(passed=True,metrics=metrics,comparison=str(root/'RESULTS.md'))))


if __name__=='__main__':main()
