from study import *
import platform,py_compile
sys.path.insert(0,str(HERE.parents[2]/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from marker_review import trc

def main():
    iv=json.loads((HERE/'input-verification.json').read_text());rv=json.loads((HERE/'results-verification.json').read_text());sv=json.loads((HERE/'threshold-sensitivity/verification.json').read_text())
    assert iv['passed'] and rv['passed'] and sv['passed'] and json.loads((HERE/'final-verification.json').read_text())['passed']
    rows=json.loads((HERE/'common176/manifest.json').read_text());lookup={m['id']:m for m in rows}
    d=dict(np.load(HERE/'common176/input.npz'));extra=dict(np.load(HERE/'threshold-sensitivity/input.npz'))
    p=pd.read_csv(HERE/'common176/person-results.csv');q=pd.read_csv(HERE/'threshold-sensitivity/person-results.csv');p=pd.concat([p,q[q.view.str.startswith('CleanDelay')]],ignore_index=True)
    summary=json.loads((HERE/'common176/summary.json').read_text());sensitivity=json.loads((HERE/'threshold-sensitivity/summary.json').read_text())
    metrics=summary['metrics']+[r for r in sensitivity['metrics'] if r['view'].startswith('CleanDelay')]
    supports={}
    for name,x in [('CleanDelay',d['CleanDelay']),('CleanDelay05',extra['CleanDelay05']),('CleanDelay15',extra['CleanDelay15'])]:
        supports[name]={'horizontal_ge1s':[str(d['ids'][i]) for i in np.flatnonzero(x[:,5]>=1)],'horizontal_ge0.5s':int(sum(x[:,5]>=.5)),'axial_ge1s':int(sum(x[:,4]>=1))}
    coefficients=[]
    for s in json.loads((HERE/'common176/selections.json').read_text()):
        if s['view']!='CleanDelay':continue
        scale,model=joblib.load(HERE/s['file']);coefficient=model.coef_/scale.scale_
        coefficients.append({k:s[k] for k in ['repeat','fold','alpha']}|{'axial_seconds_coefficient':float(coefficient[4]),'horizontal_seconds_coefficient':float(coefficient[5])})
    save('support-and-coefficients.json',{'support_counts':supports,'coefficients_conditional_not_causal':coefficients})
    # Marker trace review: ordinary detected markers, not the augmented study markers.
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
    for row,id in enumerate(['sub-397','sub-296','sub-268']):
        m=lookup[id];a=read(TURN/'raw'/m['turn_file']);path=HERE/'raw'/m['turn_file'].replace('/Kinematics/','/Markers/').replace('.mot','.trc')
        t,marks,_=trc(path);tt=np.arange(0,min(t[-1],a['time'][-1]),1/30)
        root=np.stack([np.interp(tt,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']],axis=1);progress=(root-root[0])@m['axis']
        axes[row,0].plot(tt,progress,label='골반 전진 위치',color='black')
        for k in ['LAnkle','RAnkle']:
            pp=np.stack([np.interp(tt,t,marks[k][:,j]) for j in [0,2]],axis=1)
            hh,_=motion(pp,m['axis']);axes[row,1].plot(tt,hh,label='왼발' if k=='LAnkle' else '오른발',linewidth=.8)
        axes[row,1].axhline(.1,color='black',ls=':',label='0.1 m/s')
        for k,label in [('knee_angle_l','왼쪽 무릎'),('knee_angle_r','오른쪽 무릎')]:axes[row,2].plot(a['time'],a[k],label=label,linewidth=.8)
        for col in range(3):
            ax=axes[row,col];ax.axvspan(m['turn_start'],m['turn_end'],color='grey',alpha=.15,label='회전 구간')
            for j,(l,r) in enumerate(m['tug_walk_windows']):ax.axvspan(l,r,color='royalblue',alpha=.18,label='기존 3초' if j==0 else None)
            ax.axvspan(*m['candidate_window'],color='orange',alpha=.3,label='지체 후보 1.5초')
            if m['clean_windows']!=m['observed_windows']:ax.axvspan(*m['clean_windows'][0],color='green',alpha=.2,label='앉은 후보 제외 후')
            ax.set_xlabel('기록 시작 후 시간(초)');ax.grid(alpha=.2)
        axes[row,0].set_ylabel(f'{id} · TUG {m["target"]:.2f}초\n전진 위치(m)');axes[row,1].set_ylabel('발 수평 속도(m/s)');axes[row,2].set_ylabel('무릎 굽힘(도)')
        if id=='sub-397':
            for ax in axes[row]:ax.set_xlim(8,32)
        if id=='sub-268':
            for ax in axes[row]:ax.set_xlim(12,25)
    for ax in axes[0]:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('좌표·마커 대조: 실제 정지 확정에는 원본 영상과 검사 시점 확인이 필요하다',fontsize=14)
    fig.savefig(HERE/'marker-review.png',dpi=150);plt.close(fig)
    labels={'Current':'기존 별도 보행','SameTUG':'기존 TUG 3초','CleanSpeed':'지체 포함 3초: 속도만','CleanDelay05':'지체 추가 0.05','CleanDelay':'지체 추가 0.10','CleanDelay15':'지체 추가 0.15'}
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    selected=list(labels)
    for ax,band,metric,title in [(axes[0],'all','MAE','전체 176명: 평균 절대오차'),(axes[1],'ge20','underprediction','긴 TUG 8명: 평균 과소예측')]:
        vals=[next(r[metric] for r in metrics if r['view']==v and r['band']==band) for v in selected]
        bars=ax.barh([labels[v] for v in selected],vals,color=['#aaa','#6688bb','#6688bb','#e9a15b','#e9a15b','#e9a15b']);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.19)
        ax.bar_label(bars,fmt='%.3f',padding=4);ax.set_xlabel('초 · 작을수록 좋음');ax.set_title(title)
    fig.suptitle('동일 176명·사람 분할·3초 + 같은 회전 구간 | 지체 기준에 따라 결과가 달라진다')
    fig.savefig(HERE/'comparison.png',dpi=150);plt.close(fig)
    long=p[(p.target>=20)&p.view.isin(['SameTUG','CleanSpeed','CleanDelay','CleanDelay15'])]
    long.to_csv(HERE/'long-comparison.csv',index=False)
    lines=['# OpenCap 정지·지체 정보 비교 실험 — 2026-10-31','',
      '실험과 검산을 완료했다. 놓친 지체 후보는 있었지만, 지체를 추가하면 항상 개선된다는 결과는 아니다. 주 기준 0.10 m/s에서는 악화했고, 더 넓은 저속 기준 0.15 m/s에서는 개선 가능성을 보였다. 후자는 같은 개발 자료에서 확인한 민감도 결과이며 확정 성능이 아니다.','',
      '## 비교 조건','',
      '- 42.59초인 sub-283을 학습·내부 검증·평가에서 모두 제외했다. 237명을 궤적 점검하고, 이전에 승인한 동일 TUG 보행 3초 공통 176명으로 학습했다. 긴 TUG(20초 이상) 8명이 모두 포함된다.',
      '- 기존 5가지 사람 분할 × 5개 평가 묶음과 내부 3분할을 그대로 복사했다. 같은 사람이 학습과 평가에 함께 들어가지 않는다. Ridge 설정은 내부 학습 사람의 MAE로만 선택했다.',
      '- 모든 비교는 비회전 구간 합계 3초 + 기존 회전 D초(총 4.43~9.00초)다. 지체 후보 1.5초와 기존 구간 중 겹치지 않는 1.5초를 사용했다. 같은 새 구간의 속도만 넣은 대조군을 두었다.',
      '- 지체 입력은 선택한 3초 안에서 전진 방향 속도가 낮은 시간과 전체 수평 속도가 낮은 시간이다. 전체 기록의 지체 길이·타임스탬프·TUG 총길이를 넣지 않았다.',
      '- 다만 구간 선택에는 전체 기록을 훑었다. 모델에 넣은 시간만 맞춘 실험으로, 촬영을 3초만 해도 된다는 뜻은 아니다.','',
      '## 정지인가, 추적 오류인가','',
      '- 공개 ZIP 파일 종류: MOT 5,022개, TRC 6,210개, OSIM 1,188개, YAML 1,188개. RGB 영상은 없다. 긴 TUG 8명과 추가 정지 후보 1명, 합계 9명의 TRC·메타데이터 18개 파일을 부분 다운로드하고 ZIP CRC·크기·SHA256을 확인했다.',
      '- sub-397(TUG 28.03초): 이전에 놓친 13.44~14.94초 구간에서 골반 중심의 순이동은 4.0cm, 양 발의 저속 비율은 100%, 동일 마커값이 얼어붙은 구간은 0초였다. 전후 이동과 이어지는 발·골반 궤적을 보면 실제 지체와 일관된다. 그러나 RGB가 없어 실제 정지, 안내 대기, 검사 시간 포함 여부를 확정하지 못했다.',
      '- sub-397의 11~21초 전체가 완전한 정지는 아니다. 골반 수평 저속 비율은 58%, 양 발 저속은 68.3%로, 좌우 움직임·자세 변화가 섞였다. 이 10초 길이는 점검용이며 모델에 입력하지 않았다.',
      '- sub-296(TUG 21.63초): 전진은 느려도 발이 움직이고 골반의 수평 속도 중앙값은 약 0.24m/s다. 방향이 바뀌거나 옆으로 움직인 구간을 완전 정지로 해석하면 안 된다.',
      '- sub-268(TUG 17.27초): 최초 정지 후보는 무릎 약 95도와 낮은 골반 높이로 앉은 자세에 가까웠다. 결과를 보기 전에 모든 사람에게 동일 자세 조건을 적용한 비교를 추가했다. 바뀐 사람은 이 1명이며 제외한 사람은 없다.',
      '- TRC와 MOT는 같은 OpenCap 처리 결과다. 두 자료의 일치는 독립적인 영상 검증이 아니다. 현재 QC는 큰 좌표 점프·완전 동결을 확인할 수 있으나 작은 추적 오류를 배제하지 못한다.','',
      '## 결과','',
      '아래는 앉은 자세 후보를 거른 조건이다. 과소예측은 max(실제−예측, 0)의 사람별 평균이며, 초과예측과 상쇄하지 않는다.','',
      '| 입력 | 전체 MAE | 긴 TUG MAE | 긴 TUG 과소예측 |','|---|---:|---:|---:|']
    for v in selected:
        allrow=next(r for r in metrics if r['view']==v and r['band']=='all');tail=next(r for r in metrics if r['view']==v and r['band']=='ge20')
        lines.append(f'| {labels[v]} | {allrow["MAE"]:.3f} | {tail["MAE"]:.3f} | {tail["underprediction"]:.3f} |')
    lines += ['',
      '주 기준 0.10에서는 같은 구간의 속도 대조군보다 과소예측이 0.278초 늘고 전체 MAE도 0.007초 늘었다. 긴 TUG 8명 중 4명 개선·4명 악화, 사람 분할 5가지 중 2가지에서 개선했다.',
      '',
      '0.15 기준에서는 같은 구간의 속도 대조군보다 과소예측이 0.441초 줄고 전체 MAE가 0.028초 줄었다. 과소예측 개선은 분할 5가지 중 4가지였다. 기존 TUG 3초 모델 대비로는 5가지 모두 개선했지만, 개선량의 사람별 재표집 95% 범위는 −1.09~+0.29초(같은 구간 대조군 기준)로 0을 포함한다. 8명만으로 안정적인 효과를 확정할 수 없다.',
      '',
      '0.05·0.15는 처음 정한 보조 기준이었다. 앉은 구간 교정 후 이 두 기준을 다시 확인한 실험은 주 결과를 본 뒤 수행했으며, 모두 보고했다. 가장 좋은 기준을 선택해 독립적으로 검증된 성능처럼 제시하지 않는다.',
      '',
      '## 해석과 한계','',
      '- 지체 후보를 포함하도록 구간만 바꾸면 sub-397의 과소예측은 13.45초에서 11.93초로 줄었다. 하지만 0.10 기준의 지체 특징을 더하면 14.79초로 다시 커졌다. 유용한 구간을 찾는 것과 그 특징을 학습하는 것은 별개다.',
      '- 앉은 후보를 거른 뒤 0.10 기준에서 수평 저속이 1초 이상인 사람은 sub-397 하나뿐이며, 0.5초 이상도 3명뿐이다. sub-397을 평가할 때 학습 쪽에는 같은 강도의 지체 사례가 없다. 이 희소성이 일반화가 어려운 이유로 보인다. 인과관계가 입증된 것은 아니다.',
      '- 0.15는 완전 정지뿐 아니라 느린 이동까지 포착한다. 이 넓은 기준의 개선 가능성은 남지만, 실제 정지를 정확히 검출해서 좋아졌다고 해석할 수 없다.',
      '- 임상 점수는 같은 사람·방문의 실제 TUG 값이나, 촬영된 cone/line 시행과 스톱워치 시행의 일대일 연결은 아직 미확인이다. 단일 카메라 임상 성능이나 SPPB 나머지 영역 예측을 검증한 결과가 아니다.','',
      '## 검증·재현','',
      f'- 주요 200개와 보조 50개, 총 250개 최종 모델 및 내부 학습 3,000개를 별도로 재학습해 재계산했다. 예측 차이 최대 {max(rv["cohorts"]["common176"]["max_refit_difference"],sv["max_difference"]):.1g}. 이전 두 기준 모델의 예측도 정확히 재현했다.',
      '- 입력 특징은 원본 위치에서 별도 계산하여 주 비교 차이 0을 확인했다. 사람 분할, 동일 3초, 회전 구간 비중복, 42.59초 제외, 전 사람별 5회 평가를 검사했다.',
      '- 재현 순서: 같은 원자료를 참조하는 빈 별도 작업 폴더에서 audit.py → acquire_markers.py → study.py → marker_review.py → quality_prepare.py → check_inputs.py → train.py → assess.py → check_results.py → threshold_sensitivity.py → verify_final.py → finalize.py. 기존 study 폴더를 덮어쓰지 않는 보호 조건이 있다. 그림은 생성 후 직접 열어 확인했다.',
      '- 프로토콜: protocol.json / protocol-initial.json. 예측: common176/predictions.csv, threshold-sensitivity/predictions.csv. 사람별 비교: long-comparison.csv. 마커 점검: marker-review.json/png. 검산: input-verification.json, results-verification.json, threshold-sensitivity/verification.json.',
      '',
      '자료 출처: [Zenodo 공개 자료](https://zenodo.org/records/22309771), [공식 분석 저장소](https://github.com/stanfordnmbl/NMD_Detection_Quantification).']
    (HERE/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for f in HERE.glob('*.py'):py_compile.compile(str(f),doraise=True)
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__})
    manifest={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'}
    save('artifact-manifest.json',manifest)
    print(json.dumps({'complete':True,'files_hashed':len(manifest),'final_models':250,'inner_fits':3000,'supports':supports}),flush=True)
if __name__=='__main__':main()
