from study import *
import platform,py_compile
sys.path.insert(0,str(HERE.parents[2]/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    for name in ['input-verification.json','results-verification.json','stability-verification.json']:assert json.loads((HERE/name).read_text())['passed']
    summary=json.loads((HERE/'summary.json').read_text());stability=json.loads((HERE/'stability.json').read_text());matched=json.loads((HERE/'common176-inclusion-effect.json').read_text());leave=json.loads((HERE/'exclude397-evaluation-only.json').read_text())
    def metric(cohort,view,band):return next(r for r in summary[cohort]['metrics'] if r['view']==view and r['band']==band)
    def case(cohort,view,id='sub-397'):return next(r for r in stability if r['scenario']==cohort and r['view']==view and r['id']==id)
    labels={'Speed_20':'속도만','Learned10_20':'지체 영향도 함께 학습','Add10_20':'정지 시간을 그대로 더함','Move10_20':'움직일 때 속도＋정지 더함'}
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True,constrained_layout=True)
    for ax,cohort in zip(axes,['exclude283','include283']):
        for view,color in zip(labels,['#777','#d78f42','#2d8c67','#447cbc']):
            r=case(cohort,view);ax.plot(np.arange(1,6),r['predictions_by_partition'],'o-',label=labels[view],color=color)
        ax.axhline(28.03,color='black',ls='--',label='실제 28.03초');r=case(cohort,'Add10_20')
        ax.set_title(('42.59초 사례 제외' if cohort=='exclude283' else '42.59초 사례 포함')+f'\n직접 더하기: {r["min_prediction"]:.2f}~{r["max_prediction"]:.2f}초')
        ax.set(xlabel='사람 분할 번호',ylabel='sub-397 예측 TUG(초)',xticks=[1,2,3,4,5]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('같은 관찰 구간·사람 분할에서 정지 시간을 고정해 더하면 예측 변동이 줄었다',fontsize=13)
    fig.savefig(HERE/'case397-stability.png',dpi=150);plt.close(fig)
    quality=json.loads((HERE/'case283-quality.json').read_text());m=quality['manifest'];a=read(TURN/'raw'/m['turn_file'])
    phasepath=HERE.parent/'20261031-opencap-chair/prep_event/manifest.json';phase=next(r for r in json.loads(phasepath.read_text()) if r['id']=='sub-283')
    save('case283-prior-phase-context.json',{'source_sha256':sha(phasepath),'automatic_rise_candidate':phase['phase_windows']['rise'],'core_rise':phase['core_rise'],'source':'20261031-opencap-chair/prep_event/manifest.json','not_RGB_validated':True})
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    pos=np.stack([a['pelvis_tx'],a['pelvis_tz']],axis=1);progress=(pos-pos[0])@m['axis'];axes[0].plot(a['time'],progress,color='black');axes[0].set_ylabel('전진 위치(m)')
    axes[1].plot(a['time'],a['pelvis_ty'],color='black');axes[1].set_ylabel('골반 높이(m)');axes[1].set_xlabel('기록 시작 후 시간(초)')
    for ax in axes:
        ax.axvspan(*phase['phase_windows']['rise'],color='orange',alpha=.25,label='기존 자동 기립 준비·일어서기 후보')
        ax.axvspan(*m['budgets']['20']['window'],color='royalblue',alpha=.2,label='현재 최대 20초 관찰')
        ax.axvspan(m['turn_start'],m['turn_end'],color='grey',alpha=.25,label='회전 구간');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('42.59초 사례: 보행 전 기립 준비 후보가 현재 입력 밖에 있다 · 실제 검사 포함 여부는 미확인')
    fig.savefig(HERE/'case283-context.png',dpi=150);plt.close(fig)
    lines=['# 관찰 정지 시간을 직접 더하는 TUG 모델 — 2026-10-31','',
      '실험과 검산을 완료했다. 정지 시간을 고정해 더하면 28.03초 사례의 분할별 변동이 크게 줄었다. 속도만 쓰는 모델보다 긴 TUG의 평균 과소예측과 절대오차도 감소했다. 다만 기존 지체 영향까지 학습하는 모델과 비교하면 과소예측은 늘고 과대예측은 줄었다. 모든 지표를 동시에 개선한 결과는 아니다.','',
      '## 실제 비교 방법','',
      '- 주 비교: 42.59초 sub-283 제외 176명·긴 TUG 8명. 민감도: sub-283을 복원한 177명·긴 TUG 9명. 기존 사람의 5가지 바깥 분할과 내부 3분할·관찰 구간은 바꾸지 않았다. sub-283은 원래 238명 실험의 바깥·내부 분할 소속에 복원했다.',
      '- 기존 최대 10·20·30초 관찰 구간과 저속 검출을 그대로 재사용했다. 수평 속도 0.10m/s 미만 주 기준과 0.15m/s 미만 민감도를 모두 비교했다. 옆으로 움직이는 구간까지 정지로 더하지 않도록 전진 방향 지체가 아닌 수평 저속 합계만 더했다.',
      '- 학습 정답 = 실제 임상 TUG − 관찰한 정지 후보 합계. 예측 = max(모델이 예측한 나머지 시간, 0) ＋ 관찰한 정지 후보 합계. 정지 시간의 계수는 학습하지 않고 1로 고정한다.',
      '- Add는 기존 속도 평균·변동·변화량·회전 시간 4개를 그대로 사용한다. Move는 같은 관찰 구간에서 정지 후보를 뺀 속도로 3개 속도 특징을 다시 계산한다. 두 방법 모두 같은 정지 시간을 뺀 정답을 학습하고 같은 시간을 마지막에 더한다.',
      '- Speed 대조군은 같은 4개 입력으로 전체 TUG를 바로 예측한다. Learned 대조군은 4개 속도·회전 입력에 4개 지체 길이를 더해 그 영향까지 회귀 모델이 학습한다.',
      '- Ridge SVD, alpha 0.1/1/10/100, 학습 집단만의 정규화, 내부 최종 TUG MAE에 따른 설정 선택을 유지했다. 평가할 사람의 정답으로 보정하거나 정지 계수를 선택하지 않았다.',
      '- 정답에서 뺀 값은 자동 검출한 저속 후보다. 나머지에는 기립·앉기·미관찰 지체도 남으므로 독립적으로 확인된 순수 이동시간 정답은 아니다.','',
      '## 주 비교: 42.59초 제외, 최대 20초','',
      '| 방법 | 전체 MAE | 긴 TUG MAE | 긴 TUG 과소예측 | 긴 TUG 과대예측 |','|---|---:|---:|---:|---:|']
    for view,label in labels.items():
        allrow=metric('exclude283',view,'all');tail=metric('exclude283',view,'ge20')
        lines.append(f'| {label} | {allrow["MAE"]:.3f} | {tail["MAE"]:.3f} | {tail["underprediction"]:.3f} | {tail["overprediction"]:.3f} |')
    lines += ['',
      '단위는 초다. 각 사람이 받는 5개 평가 예측의 오차를 먼저 평균한 뒤 사람 간 평균했다. 과소·과대예측은 서로 상쇄하지 않는다.',
      '',
      '- Add는 Speed 대비 긴 TUG 과소예측 6.108→5.304초, MAE 6.185→5.447초로 개선했다. 분할 5가지 모두 평균은 개선했지만 사람별로는 8명 중 2명만 개선했다. 과소예측 차이 −0.804초의 조건부 95% 구간은 −2.529~+0.134초로 0을 포함한다.',
      '- Add는 Learned 대비 긴 TUG MAE 5.787→5.447초, 과대예측 0.962→0.143초로 개선했다. 반면 과소예측 4.825→5.304초와 전체 MAE 1.055→1.087초는 증가했다.',
      '- 정지 구간을 제외한 속도로 학습하는 Move는 Add보다 주 비교에서 추가 이득이 없었다. 사후에 더 좋은 모델을 골라 주 모델을 바꾸지 않았다.',
      '',
      '## 28.03초 사례의 안정성','',
      '| 42.59초 포함 여부 | 방법 | 예측 평균 | 예측 최소~최대 | 분할 간 표준편차 | 개인 MAE |','|---|---|---:|---:|---:|---:|']
    for cohort in ['exclude283','include283']:
        for view in labels:
            r=case(cohort,view);lines.append(f'| {"제외" if cohort=="exclude283" else "포함"} | {labels[view]} | {r["mean_prediction"]:.3f} | {r["min_prediction"]:.3f}~{r["max_prediction"]:.3f} | {r["prediction_SD"]:.3f} | {r["MAE"]:.3f} |')
    lines += ['',
      '주 Add의 분할 간 표준편차는 12.923→0.349초로 약 97.3% 감소했다. 다만 예측 평균 23.14초는 실제 28.03초보다 약 4.89초 짧다. 표준편차는 동일 사람의 5개 분할 간 변동이며, 새 환자에 대한 임상 신뢰구간이 아니다.',
      '',
      '## 42.59초 사례를 제외해야 하는가','',
      '값이 크다는 이유만으로 계속 제외할 근거는 확인되지 않았다. 이 사례의 현재 TUG 보행 3초 선택은 기존 품질 조건을 통과했고, 10·20·30초 상한에서 사용한 보행 입력의 추적 급변 제외 시간은 0초였다. 예전 별도 10m 보행의 첫 부분에 있었던 급변을 현재 TUG 기록 전체의 오류로 확대해서는 안 된다.',
      '',
      '현재 판단은 품질 기준을 동일하게 적용한 전체 자료의 결과와, 이 사례를 제외한 민감도 결과를 함께 보고하는 것이다. 이 비교에서는 이전 결과와 연결하기 위해 제외 조건을 주 비교로 유지했다. 실제 라벨 오류·연결 오류가 확인되면 그 근거로 수정하거나 제외할 수 있다. 큰 값이라는 이유로 자동 삭제하지 않는 원칙은 [NIST 이상치 안내](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)와도 부합한다.',
      '',
      '| 177명 포함 실험 | 전체 MAE | 긴 9명 MAE | 긴 9명 과소예측 |','|---|---:|---:|---:|']
    for view,label in labels.items():
        arow=metric('include283',view,'all');tail=metric('include283',view,'ge20');lines.append(f'| {label} | {arow["MAE"]:.3f} | {tail["MAE"]:.3f} | {tail["underprediction"]:.3f} |')
    lines += ['',
      '위 9명 평균을 176명 실험의 8명 평균과 직접 비교하면 평가 구성도 달라진다. 같은 기존 8명만 평가 집계에 남기면, 포함 실험의 Add는 긴 MAE 4.917초·과소예측 4.385초다. 포함 실험의 Learned는 각각 6.470초·5.147초다. sub-283 복원 효과와 모델 간 비교를 구분한 결과는 common176-inclusion-effect.json에 저장했다.',
      '',
      'sub-283 자체의 Add 예측은 평균 25.03초로 실제 42.59초보다 17.56초 짧았다. 현재 관찰한 보행·회전 구간에서 정지 후보는 0초다. 이전 자동 분할의 기립 준비·일어서기 후보는 기록 7.20~20.63초(13.43초)이고, 현재 보행 관찰은 22.77초 이후부터 시작한다. 입력 밖에 긴 동작 후보가 있다는 뜻이며, 그 시간이 실제 임상 스톱워치에 어떻게 포함됐는지까지 확인한 것은 아니다.','',
      '## 관찰시간·검증 한계','',
      '- 최대 20초 조건의 기존 긴 8명은 실제 13.33~20초를 관찰했다. 최대 30초 상한도 기존 176명에서는 실제 최장 21.33초, sub-283 포함 시 최장 26.33초다. 실제 30초 관찰 효과를 검증한 것은 아니다.',
      '- 주 Add의 제외176 조건에서 긴 과소예측은 최대10초 6.067, 최대20초 5.304, 최대30초 상한 5.284초였다. 0.15m/s 민감도의 최대20초에서는 5.273초였다. 전 조건 수치와 사람별 결과는 각 cohort의 metrics.csv·person-results.csv에 보존했다.',
      '- 기존 긴 8명에서 sub-397만 평가 집계에서 빼면 Add의 Speed 대비 이득은 사라진다. 작은 표본에서 개선이 특정 정지 사례에 집중된 한계가 남아 있다. 해당 사람을 빼고 재학습한 결과는 아니다.',
      '- 공개 RGB 영상과 정확한 임상 스톱워치 시행의 일대일 연결은 여전히 미확인이다. 자동 저속 후보가 실제 정지인지, 안내 대기인지 확정할 수 없다. 단일 카메라 임상 검증 또는 SPPB 점수 검증이 아니다.',
      '- 전체 기록으로 관찰 구간·회전 경계를 골랐으므로 실시간 10~30초 촬영 시스템을 검증한 결과가 아니다.',
      '',
      '## 검산과 재현','',
      '- 1,050개 최종 모델과 12,600개 내부 학습을 별도로 다시 학습해 예측과 설정 선택을 재현했다. 176명의 기존 9개 대조 조건 예측을 정확히 재현했다. 정지 시간 차감·고정 덧셈·잔여시간 하한을 확인했다.',
      '- 움직일 때 속도 입력을 원본 좌표에서 별도 계산했고 차이는 0이었다. 관찰시간·사람 분할 유지, sub-283의 원래 분할 복원, 28.03초 사례의 표준편차·범위, 같은 176명 포함 효과와 조건부 구간을 독립 검산했다.',
      '- 실행 순서: study.py → check_inputs.py → train.py → assess.py → check_results.py → stability.py → verify_stability.py → finalize.py. 이전 결과를 덮어쓰지 않는 보호 조건이 있다. 새 패키지나 GPU는 사용하지 않았다.',
      '- 핵심 코드: study.py의 fit. 학습할 때 y에서 관찰 정지 시간을 빼고, 예측할 때 같은 시간을 더한다. Move의 속도 정의 변경은 moving_features에 분리했다.',
      '- 주요 근거: protocol.json, case283-quality.json, exclude283/predictions.csv, include283/predictions.csv, stability.json, common176-inclusion-effect.json, input-verification.json, results-verification.json, stability-verification.json.','']
    (HERE/'STATUS.md').write_text('\n'.join(lines),encoding='utf8')
    for f in HERE.glob('*.py'):py_compile.compile(str(f),doraise=True)
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__})
    save('artifact-manifest.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'})
    print('REPORT AND FIGURES SAVED; 1050 final models and12600 inner fits verified',flush=True)
if __name__=='__main__':main()
