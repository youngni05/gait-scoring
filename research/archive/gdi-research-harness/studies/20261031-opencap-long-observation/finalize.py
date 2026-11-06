from study import *
import platform,py_compile
sys.path.insert(0,str(HERE.parents[2]/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    assert json.loads((HERE/'input-verification.json').read_text())['passed'];verification=json.loads((HERE/'results-verification.json').read_text());assert verification['passed']
    summary=json.loads((HERE/'summary.json').read_text());coverage=json.loads((HERE/'coverage.json').read_text());out=HERE/'capped176';d=np.load(out/'input.npz')
    people=pd.read_csv(out/'person-results.csv');pred=pd.read_csv(out/'predictions.csv');plans={(r['repeat'],f['fold']):f for r in json.loads((out/'splits.json').read_text()) for f in r['splits']}
    rows=json.loads((out/'manifest.json').read_text());diagnostic=[]
    for s in json.loads((out/'selections.json').read_text()):
        if s['view']!='Delay10_20':continue
        f=plans[s['repeat'],s['fold']];scale,model=joblib.load(HERE/s['file']);x=d[s['view']]
        for i in f['test']:
            if d['y'][i]<20:continue
            trainmax=x[f['train'],4:].max(0)
            diagnostic.append({'id':str(d['ids'][i]),'repeat':s['repeat'],'fold':s['fold'],'alpha':s['alpha'],'delay_features':x[i,4:].tolist(),'training_delay_maxima':trainmax.tolist(),'outside_training_range':(x[i,4:]>trainmax).tolist(),
                'prediction':float(model.predict(scale.transform(x[i:i+1]))[0]),'scaled_delay_contributions':(scale.transform(x[i:i+1])[0,4:]*model.coef_[4:]).tolist()})
    save('long-training-support.json',diagnostic)
    long=people[people.target>=20].copy();long.to_csv(HERE/'long-comparison.csv',index=False)
    excluded=[]
    for v in ['SameTUG','Speed_20','Delay10_20','Delay15_20']:
        z=long[(long.view==v)&(long.id!='sub-397')]
        excluded.append({'view':v,'people':len(z),**{k:float(z[k].mean()) for k in ['MAE','underprediction','overprediction']}})
    save('exclude397-evaluation-only.json',{'training_unchanged':True,'metrics':excluded,'scope':'evaluation aggregation only, NOT retraining without397'})
    # Plot the tradeoff explicitly: lower underprediction is not sufficient if overprediction rises.
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    labels={'SameTUG':'기존 보행 3초+회전','Speed_10':'최대 10초: 속도','Delay10_10':'최대 10초: 지체 추가','Speed_20':'최대 20초: 속도','Delay10_20':'최대 20초: 지체 추가','Delay10_30':'최대 30초: 지체 추가*'}
    metrics=summary['capped176']['metrics'];fig,axs=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,band,key,title in [(axs[0],'all','MAE','전체 176명 절대오차'),(axs[1],'ge20','underprediction','긴 TUG 8명 과소예측'),(axs[2],'ge20','overprediction','긴 TUG 8명 과대예측')]:
        vals=[next(r[key] for r in metrics if r['view']==v and r['band']==band) for v in labels]
        bars=ax.barh(list(labels.values()),vals,color=['#888','#6688bb','#df9954','#6688bb','#df9954','#df9954']);ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_xlim(0,max(vals)*1.25);ax.set_title(title);ax.set_xlabel('초 · 작을수록 좋음')
    fig.suptitle('관찰 확대와 지체 길이의 효과 | *실제 최대 기록은 21.33초, 30초 관찰 검증은 아님',fontsize=13)
    fig.savefig(HERE/'comparison.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);m=next(m for m in rows if m['id']=='sub-397');a=read(TURN/'raw'/m['turn_file'])
    pos=np.stack([a['pelvis_tx'],a['pelvis_tz']],axis=1);progress=(pos-pos[0])@m['axis'];axs[0].plot(a['time'],progress,color='black')
    for b,color in [(20,'orange'),(10,'royalblue')]:axs[0].axvspan(*m['budgets'][str(b)]['window'],color=color,alpha=.2,label=f'최대 {b}초 관찰')
    for i,e in enumerate(m['budgets']['20']['events']['10']['horizontal']):axs[0].axvspan(e['start'],e['end'],color='green',alpha=.4,label='수평 저속 지체' if i==0 else None)
    axs[0].set(xlabel='기록 시작 후 시간(초)',ylabel='전진 위치(m)',title='sub-397: 긴 관찰에서 지체를 포착');axs[0].legend()
    for v,color in [('SameTUG','#888'),('Speed_20','#6688bb'),('Delay10_20','#df9954')]:
        z=pred[(pred.id=='sub-397')&(pred.view==v)].sort_values('repeat');axs[1].plot(z['repeat']+1,z.prediction,'o-',label=labels[v],color=color)
    axs[1].axhline(28.03,color='black',ls='--',label='실제 TUG 28.03초');axs[1].set(xlabel='사람 분할 번호',ylabel='예측 TUG(초)',title='지체를 포착해도 예측은 분할에 따라 크게 변함',xticks=[1,2,3,4,5]);axs[1].legend(fontsize=9)
    fig.savefig(HERE/'case397.png',dpi=150);plt.close(fig)
    lines=['# 관찰시간 확대와 정지·지체 지속시간 비교 — 2026-10-31','',
      '실험과 독립 검산을 완료했다. 최대 20초까지 관찰하고 지체의 총길이·최장길이를 입력하면 평균 과소예측은 줄었다. 그러나 개선이 한 사례에 크게 의존하고 일부 분할에서 과대예측이 커져, 긴 TUG 문제를 안정적으로 해결했다고 볼 수 없다.','',
      '## 자료와 비교 조건','',
      '- 기존 176명과 긴 TUG(20초 이상) 8명을 유지하고 42.59초 sub-283을 모든 학습·내부 선택·평가에서 제외했다. 기존 5가지 사람 분할과 내부 3분할을 그대로 유지했다.',
      '- 기존 회전과 그 앞뒤 보행을 포함한 채, 먼저 앞쪽으로 관찰을 늘리고 보행 시작점에 도달하면 뒤쪽으로 늘렸다. 최대 10·20·30초이며 원본 끝에서 멈췄다. 짧은 기록을 반복·보간 확대하거나 부족한 시간을 0으로 채우지 않았다.',
      '- 보행 시작 이후 남은 기록이 10초 이상인 사람은 72명(긴 8명 모두 포함), 20초 이상은 3명, 30초는 0명이었다. 정확한 10초 비교는 72명으로도 별도 재학습했다. 20초·30초를 모든 사람에게 실제 관찰한 실험이 아니다.',
      '- 긴 구간의 속도 평균·변동·인접 변화량과 기존 회전 시간을 쓰는 대조군을 두었다. 지체 모델은 동일 구간에서 전진 방향·수평 속도가 낮은 구간의 합계와 최장 길이 4개를 추가했다.',
      '- 주 기준은 0.10m/s, 사전 지정 민감도 기준은 0.15m/s다. 0.2초 이하의 짧은 끊김을 연결하고 0.5초 이상 이어진 저속 구간을 센다. 회전·앉은 자세·추적 이상 구간을 가로질러 연결하지 않는다.',
      '- 원본 좌표의 6m/s 초과 급변 주변 0.15초를 제외했다. 보행에 해당하는 입력에서 영향을 받은 사람은 sub-268과 sub-030이며, 사람 자체는 제외하지 않았다. 앉은 자세는 이전 높이·무릎·엉덩관절 조건으로 걸렀다.',
      '- Ridge 설정 후보·학습 사람 내부 정규화·내부 MAE 선택·예측 하한 0을 유지했다. 정답 TUG, 전체 녹화 길이, 관찰 구간의 타임스탬프를 입력에 넣지 않았다. 구간 선택은 전체 시계열을 참조하므로 실시간 검증이 아니다.','',
      '## 실제 관찰시간','',
      '| 상한 | 상한을 채운 사람 | 실제 최소 / 중앙 / 최대 |','|---|---:|---:|']
    for b,r in coverage.items():lines.append(f'| {b}초 | {r["full_budget"]}명 | '+ ' / '.join(f'{v:.2f}' for v in r['observed_min_med_max'])+'초 |')
    longtimes=[m['budgets']['20']['observed_seconds'] for m in rows if m['target']>=20]
    lines += ['',f'긴 TUG 8명의 최대 20초 조건은 실제 {min(longtimes):.2f}~{max(longtimes):.2f}초다. 대부분의 짧은 TUG는 10초 상한에 도달하기 전에 기록이 끝난다.','',
      '## 동일 176명 결과','',
      '단위는 초다. 모든 지표는 사람별로 5회 평가의 오차를 평균한 뒤 사람 간 평균했다. 과소예측과 과대예측은 서로 상쇄하지 않는다.','',
      '| 입력 | 전체 MAE | 긴 TUG MAE | 긴 TUG 과소예측 | 긴 TUG 과대예측 |','|---|---:|---:|---:|---:|']
    order=['Current','SameTUG','Speed_10','Delay10_10','Delay15_10','Speed_20','Delay10_20','Delay15_20','Speed_30','Delay10_30','Delay15_30']
    for v in order:
        allrow=next(r for r in metrics if r['view']==v and r['band']=='all');tail=next(r for r in metrics if r['view']==v and r['band']=='ge20')
        lines.append(f'| {v} | {allrow["MAE"]:.3f} | {tail["MAE"]:.3f} | {tail["underprediction"]:.3f} | {tail["overprediction"]:.3f} |')
    lines += ['',
      'Current는 기존 별도 보행 3초+회전, SameTUG는 기존 같은 TUG 보행 3초+회전이다. Speed는 해당 상한까지 관찰한 속도 대조군이다. Delay10/15는 각각 0.10/0.15m/s 기준의 지체 특징을 더한 모델이다. 이름 뒤 숫자는 관찰시간 상한이다.',
      '',
      '- 최대 20초 주 모델은 같은 시간 속도 대조군 대비 전체 MAE 1.159→1.055초, 긴 TUG MAE 6.185→5.787초, 과소예측 6.108→4.825초로 개선했다. 과소예측은 분할 5개 모두 개선, 8명 중 6명 개선이다. 그러나 과대예측은 0.077→0.962초로 증가했다.',
      '- 같은 시간 대조군 대비 과소예측 차이 −1.283초의 조건부 사람 재표집 95% 범위는 −3.619~+0.145초로 0을 포함한다. 반복 분할은 같은 8명의 재평가이며 독립 사례가 늘어난 것이 아니다.',
      '- 10초에서 지체 추가는 주 기준상 도움이 되지 않았다. 20초와 최대 30초 결과는 거의 같았고, 실제 30초 기록이 없어 30초의 추가 효과를 검증할 수 없었다.',
      '- 정확히 10초를 확보한 72명에서도 주 지체 모델은 최대 10초 과소예측 6.510초, 최대 20초 5.094초였다. 같은 최대 20초 속도 대조군은 6.111초였다. 긴 8명은 동일하므로 독립적인 재현 결과는 아니며, 176명 수치와 직접 비교하면 안 된다.','',
      '## 개선이 한 사람에 의존하는가','',
      '- sub-397(실제 28.03초)의 최대 20초 입력에서 전진 지체는 합계/최장 모두 약 8.97초, 수평 저속은 합계 6.83초·최장 3.47초였다. 회전·앉기·관측 밖 시간은 포함하지 않았다. 이 값은 자동 검출 후보 시간이며 임상적으로 확정한 정지 시간이 아니다.',
      '- 이 사람의 기존 같은 TUG 짧은 입력 예측 평균은 14.58초, 긴 속도만 모델은 16.40초, 지체 추가 모델은 31.87초다. 하지만 지체 추가 모델의 분할별 예측은 20.26~56.62초로 크게 흔들린다. 평균만 보면 좋은 근접값처럼 보일 수 있으나 개인별 5회 MAE는 9.16초다.',
      '- sub-397만 평가 집계에서 빼면 나머지 긴 7명의 기존 짧은 모델 대비 최대 20초 지체 모델은 과소예측이 약 0.234초, MAE가 약 0.216초 증가한다. 이는 평가 민감도이며 이 사람을 빼고 재학습한 실험은 아니다.',
      '- 이 사람을 평가할 때 학습 집단의 수평 지체 합계 최댓값은 분할에 따라 0~1.63초였다. 평가 입력의 6.83초는 이를 크게 벗어났다. 새 입력을 확보했어도 그 범위에서의 점수 관계를 학습할 사람이 부족했다.',
      '- 0.15m/s 민감도에서도 최대 20초 결과는 전체 MAE 1.042초·긴 과소예측 4.765초지만 sub-397 예측은 18.08~57.79초로 불안정했다. 가장 좋은 설정만 선택해 확정 성능으로 보고하지 않는다.','',
      '0.15m/s에서는 sub-397을 평가 집계에서 뺀 나머지 7명도 기존 짧은 입력 대비 MAE 약 0.149초·과소예측 약 0.187초가 개선했다. 따라서 모든 조건의 개선이 전적으로 한 사람뿐이라고 단정하지 않는다. 주 기준에서 개선이 사라지고 기준에 따라 결과가 달라진다는 점이 한계다.','',
      '## 확인 범위와 검산','',
      '- 550개 최종 모델과 6,600개 내부 학습을 별도로 재학습해 예측·설정 선택을 재현했다. 원본 속도와 지체 길이를 별도 코드로 재계산했고, 분할·원본 해시·실제 관찰 범위·지표·조건부 구간을 확인했다. 기존 176명 기준 두 모델 예측도 재현했다.',
      '- 공개 자료의 원본 RGB 영상과 실제 스톱워치 시행의 연결은 여전히 없다. 앞선 9명의 마커 점검이 영상 검증을 대신하지 않는다. 안내 대기를 지체로 오인할 가능성도 남아 있다.',
      '- 관찰이 길어지면 같은 TUG 동작의 상당 부분을 직접 보게 된다. 보행 능력으로 별도 검사를 예측하는 문제에서, 관찰한 동작 시간을 자동 측정하는 문제 쪽으로 가까워진다. 두 연구 목적을 구분해야 한다.',
      '- 실행 순서: scout.py → study.py → check_inputs.py → train.py → assess.py → check_results.py → finalize.py. 기존 결과 덮어쓰기 보호 조건이 있으므로 재실행은 빈 별도 결과 폴더에서 한다.',
      '- 주요 파일: protocol.json, coverage.json, capped176/predictions.csv, exact10/predictions.csv, long-comparison.csv, long-training-support.json, exclude397-evaluation-only.json, input-verification.json, results-verification.json.','']
    (HERE/'STATUS.md').write_text('\n'.join(lines),encoding='utf8')
    for f in HERE.glob('*.py'):py_compile.compile(str(f),doraise=True)
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__})
    save('artifact-manifest.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'})
    print(json.dumps({'verified_final_models':verification['models'],'verified_inner_fits':verification['inner_fits'],'case397':[r for r in diagnostic if r['id']=='sub-397'],'eval_without397':excluded}),flush=True)
if __name__=='__main__':main()
