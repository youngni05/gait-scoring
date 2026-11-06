from study import *
sys.path.insert(0,str(HERE.parents[1]/'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    v=json.loads((HERE/'verification.json').read_text());assert v['passed'] and v['models']==450 and v['inner_fits']==5400
    cases=json.loads((HERE/'case-timing-audit.json').read_text());sources=json.loads((HERE/'marker-boundaries.json').read_text());summaries=json.loads((HERE/'summary.json').read_text());predictions=json.loads((HERE/'case-predictions.json').read_text())
    assert json.loads((HERE/'marker-sensitivity-verification.json').read_text())['passed']
    tail=json.loads((HERE/'marker-tail-sensitivity.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for axs,r in zip(axes,sorted(cases,key=lambda r:r['id'])):
        m=next(m for m in json.loads((PRIOR/'include283/manifest.json').read_text()) if m['id']==r['id']);a=read(TURN/'raw'/m['turn_file'])
        src=next(z for z in sources if z['id']==r['id'] and 'tug_cone' in z['file']);fps,t,names,xyz=trc(HERE/src['file']);h=(xyz[:,list(names).index('RHJC_study'),1]+xyz[:,list(names).index('LHJC_study'),1])/2
        ax=axs[0];ax.plot(a['time'],a['pelvis_ty'],label='MOT pelvis',lw=1);ax.plot(t,h,label='TRC hip-center mean',lw=.8,alpha=.8)
        for w,color,label in [([r['candidate_start'],r['rise_end']],'#d95f02','Automatic prep + rise'),([r['core_rise_start'],r['rise_end']],'#1b9e77','Core rise'),([r['sit_start'],r['sit_end']],'#7570b3','Sit candidate')]:ax.axvspan(*w,color=color,alpha=.2,label=label)
        ax.axvline(src['author_height_proxy']['start'],color='#e7298a',ls='--',label='Author height-proxy start')
        if r['id']=='sub-283':
            revised=next(z for z in tail if z['id']==r['id'] and z['removed_tail_frames']==1)
            ax.axvline(revised['boundary']['start'],color='#a6761d',ls=':',label='Same proxy without last frame')
        ax.set_title(f'{r["id"]} | clinical TUG {r["target"]:.2f}s');ax.set_ylabel('Height (m)');ax.set_xlabel('Recording time (s)');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
        ax=axs[1]
        for k in ['pelvis_tilt','lumbar_extension','hip_flexion_l','hip_flexion_r']:ax.plot(a['time'],a[k],lw=.8,label=k)
        ax.axvspan(r['candidate_start'],r['core_rise_start'],color='#d95f02',alpha=.15);ax.axvline(r['core_rise_start'],color='#1b9e77',ls='--');ax.set_xlim(0,r['rise_end']+1);ax.set_xlabel('Recording time (s)');ax.set_ylabel('Angle (deg)');ax.set_title('Pre-rise posture change is not a stopwatch start');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.suptitle('Clinical start signal unknown for both cases; model-derived boundaries only')
    fig.savefig(HERE/'boundary-comparison.png',dpi=150);plt.close(fig)
    source_notes=[{'url':'https://zenodo.org/records/22309771','finding':'public415person release lists kinematics, markers, models and metadata; local ZIP index contains no media or timing annotation file names'},
      {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC13648720/','finding':'Methods:415person protocol follows Ruth et al; released session dictionaries describe cameras and processing, not stopwatch events; preprint'},
      {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12416922/','finding':'Data collection:stopwatch measurements simultaneous with OpenCap by trained evaluators; algorithmic height-based boundaries differ from clinical chair-contact times; not per-person timestamp release'},
      {'url':'https://raw.githubusercontent.com/stanfordnmbl/opencap-fshd-dm-analysis/main/feature_extraction/featurize_tug.py','finding':'normalized hip center height30percent start proxy; algorithmic feature not stopwatch annotation; code reproduced on targetTRC'},
      {'url':'https://pmc.ncbi.nlm.nih.gov/articles/instance/12416922/bin/NIHMS2107850-supplement-appendix.pdf','finding':'web access returned browser verification page; supplement body not inspected; not used as evidence'}]
    save('source-review.json',source_notes)
    lines=['# TUG 계측 시작·준비시간 확인 — 공개 자료 감사와 보수적 재실험','',
      '**실제 계측 시작 시점은 두 사례 모두 확인하지 못했다.** 공개 원자료·연구 방법·코드를 추가 확인했고, 사용자도 별도 영상·계측 로그가 없다고 답했다. 따라서 확인된 준비시간만 사용하는 본실험은 필요한 정답 주석이 없어 미실행이다. 자동 후보를 임상적으로 확인된 시간으로 바꾸지 않았다.','',
      '대신 기존82/83명·같은 사람 분리·같은 관찰 구간에서 미확인 준비 및 기립 시간을 전부 반영하지 않는 보수적 민감도 분석을 재실행했다. 이 조건에서는 앞선 평균 개선이 유지되지 않았다. 실제 준비시간이0초라는 뜻도, 준비 동작이 무용하다는 뜻도 아니다. 이전 개선은 자동 후보를 시간으로 반영하는 가정에 의존한다.','',
      '## 추가 확인한 근거','',
      '- 공개 ZIP 인덱스19,451항목: TRC6,210·MOT5,022·YAML1,188·OSIM1,188개와 디렉터리5,843개. 영상·음성 파일0개, 파일명 기준 계측 이벤트·시작 주석 후보0개.',
      '- 두 사례의 현재 세션 YAML을 새로 확보해 본문을 확인했다. 카메라·보정·마커 증강·모델 설정만 있고 시작 신호·스톱워치·시행별 계측값은 없다.',
      '- sub-283 visit-197의 TUG-cone TRC, sub-288 visit-0 ses-2의 TUG-cone 및 TUG-line TRC를 선택 다운로드하고 ZIP CRC와SHA-256을 검증했다.',
      '- 임상표의 subid·visit·tft_tug를 다시 대조했다. sub-283의visit-197은42.59초,visit-0은24.19초다. 현재 비교에서 방문을 바꾸지 않았다. sub-288 visit-0은9.15초다. 임상표에는 시행ID·녹화시각 기준 타이머 시작/종료 열이 없다.',
      '- sub-288의 별도 TUG-line 파일은6.525초이고 연구진 높이 기준으로 착석 양끝을 식별하지 못했다. 이것을9.15초의 정답 시행으로 바꾸지 않았다.',
      '- 연구진 공개 코드 저장소의 파일 목록에서도 개인별 계측 시작 주석을 찾지 못했다. 연구진 코드의 높이 기반 TUG 추정은 다시 실행해 확인했으나 스톱워치 주석은 아니다.','',
      '[원 연구 방법](https://pmc.ncbi.nlm.nih.gov/articles/PMC12416922/)은 OpenCap 촬영과 스톱워치 계측이 동시에 이루어졌으며 임상적 의자 접촉 시점과 알고리즘의 높이 기준이 다를 수 있다고 설명한다. [415명 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC13648720/)는 이 수집 절차를 따른다고 한다. 이런 일반 절차만으로 두 사람의 실제 시작 프레임을 정할 수는 없다.','',
      '## 두 사례의 경계 비교','',
      '아래 시각은 녹화 시작을0초로 한 자동 좌표 경계다. 실제 검사 시작 신호가 아니다.','',
      '| 사례 | 준비 후보 시작 | 핵심 기립 시작 | 기립 끝 | 착석 끝 후보 | 기립 시작→착석 끝 | 준비 시작→착석 끝 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(cases,key=lambda r:r['id']):lines.append(f'| {r["target"]:.2f}초 ({r["id"]}) | {r["candidate_start"]:.2f} | {r["core_rise_start"]:.2f} | {r["rise_end"]:.2f} | {r["sit_end"]:.2f} | {r["core_to_sit_seconds"]:.2f} | {r["candidate_to_sit_seconds"]:.2f} |')
    lines+=['','9.15초 사례에서는 핵심 동작 구간9.00초가 임상값에 가깝다. 준비 후보 전체를 포함하면14.50초로5.35초 길어진다. 앞의 자세 변화가 계측 전 동작일 가능성을 지지하지만, 동일 시행·착석 경계가 맞다는 조건부 해석이다. 명령 신호 없이 포함 여부를 확정하지 못했다.','',
      '42.59초 사례는 핵심 구간27.60초로14.99초 부족하고 준비 후보를 포함해도39.47초로3.12초 부족하다. 준비·지체가 중요할 가능성과 계측 시작·시행 대응 문제를 구분할 수 없다.','',
      '참고로 착석 끝에서 정답을 빼면42.59초 사례는4.08초,9.15초 사례는7.12초가 나온다. 이는 임상 정답을 사용한 역산이며 실측 시작점이 아니다. 이 시점으로 준비시간을 정하면 평가 정답이 입력으로 새므로 모델에 사용하지 않았다.','',
      '연구진의30% 높이 기준을 TRC에 적용한 시작·종료는42.59초 사례7.40~46.55초(39.15초),9.15초 사례7.71~15.82초(8.11초)였다. 같은 카메라에서 유래한 마커와MOT의 비교이므로 독립 센서 검증이 아니다. 높이 기준이 일치하거나 임상값에 가까워도 실제 시작 명령을 확인한 것으로 취급하지 않았다.','',
      '**마지막 프레임 민감도:** sub-283 TRC의 마지막 골반 높이는0.491m로 직전0.668m보다 급락하며 환산 수직 속도는10.60m/s다. 추적 오류를 의심할 강한 징후다. 이 한 프레임(1/60초)을 제외하면 연구진 전체 높이 범위 기준의 시작이7.40→19.75초, 추정 소요시간이39.15→26.27초로 바뀐다. 마지막0.1초·0.5초를 제외해도 같은 경계다. sub-288은 동일한 절차에서 경계가 변하지 않는다. 따라서 원 코드가7.40초를 시작으로 잡았다는 사실을 준비시간의 독립 확인 근거로 쓰지 않았다. 이 입력 진단은 기존 모델이나 자동 준비 구간을 바꾸지 않는다.','',
      '## 실행한 보수적 비교','',
      '기존 관찰 상한10·20·30초, 같은82/83명, 바깥5가지 사람 분할×5묶음·내부3분할을 그대로 썼다. 개인별 실제 입력 초수와구간도 이전 실험과 같다. 새 구간이나 유리한 사람을 고르지 않았다.','',
      '- Gait: 이전 보행 구간을 유지하고 보행 저속 후보만 더함.',
      '- Withheld: 이전 준비+보행 구간을 그대로 관찰하지만, 계측 포함이 미확인인 준비·기립 길이는 특징이나 덧셈에 쓰지 않음. 보행 저속 후보만 더함.',
      '- Candidate: 같은 준비+보행 구간에서 자동 후보 길이를 모두 더함. 이전 개선을 재현하는 대조 조건이다.',
      '- 이전 Reallocated는 이번 Withheld와 수학적으로 같다. 새 임상 검증처럼 세지 않고, 기존 조건을 재학습·재검산한 명시적 민감도 분석으로 보고한다.',
      '- 학습은 기존 Ridge와 alpha0.1/1/10/100을 유지했다. 내부 학습 사람의 최종TUG MAE로만 설정을 선택했다. 평가 정답을 구간·시간·보정에 쓰지 않았다.','',
      '최대20초 조건, 단위는 초다.','',
      '| 집단 | 방법 | 전체 절대오차 | 긴 사람 절대오차 | 긴 사람 과소예측 |','|---|---|---:|---:|---:|']
    for c,label in [('exclude283','42.59 제외82명 / 긴6명'),('include283','포함83명 / 긴7명')]:
        for kind,name in [('Gait','보행 정지만'),('Withheld','구간 재배분·미확인 시간 미반영'),('Candidate','자동 준비 후보 모두 반영')]:
            a=next(r for r in summaries[c]['metrics'] if r['view']==kind+'_20' and r['band']=='all');z=next(r for r in summaries[c]['metrics'] if r['view']==kind+'_20' and r['band']=='ge20');lines.append(f'| {label} | {name} | {a["MAE"]:.3f} | {z["MAE"]:.3f} | {z["underprediction"]:.3f} |')
    lines+=['','미확인 시간을 보류한 조건은 보행만 보던 기준보다 더 나빴다. 준비 구간을 확보하면서 보행 일부를 덜 관찰했지만 그 시간 정보를 사용하지 않았기 때문이다. 이 비교로 “검증된 준비시간의 효과가 없다”라고 판단할 수는 없다. 검증된 준비시간 자체가 없기 때문이다.','',
      '## 사례별 예측','',
      '| 집단 | 사례 | 방법 | 예측 평균 | 분할 표준편차 | 절대오차 |','|---|---|---|---:|---:|---:|']
    for c,id in [('include283','sub-283'),('include283','sub-288'),('exclude283','sub-288'),('exclude283','sub-397')]:
        for kind in ['Gait','Withheld','Candidate']:
            r=next(r for r in predictions if r['cohort']==c and r['id']==id and r['view']==kind+'_20');lines.append(f'| {c} | {r["target"]:.2f}초 | {kind} | {r["mean"]:.2f} | {r["sd"]:.2f} | {r["MAE"]:.2f} |')
    lines+=['','## 현재 판단과 필요한 자료','',
      '앞선 개선은 자동 준비 후보가 예측에 유용할 수 있음을 보여주는 탐색 결과로 유지한다. 그러나 그 시간을 실제 검사에 포함된 시간으로 해석하는 근거는 아직 확보되지 않았다. 특히9.15초 반례는 이 가정이 틀릴 수 있음을 보여준다. 점수가 더 잘 맞는다는 이유로 후보를 확정하거나 특정 사람만 준비시간을 빼는 규칙을 만들지 않았다.','',
      '요청한 본실험에 필요한 것은 (1)현재공개 파일과 임상 tft_tug의 정확한 시행 대응, (2)녹화시각 기준 검사 시작/종료 신호, (3)그 신호 안에 포함된 준비·기립의 경계다. 두 사례의 자료가 확보되면 사례 진단을 할 수 있고, 같은82/83명 본실험을 위해서는 그 집단에 일관된 기준의 확인이 필요하다. 요청 없이 연구진에게 메시지를 보내지 않았다.','',
      'timing-annotations.json의83명 모두를 unverified로 기록하고 시간은null로 남겼다. confirmed-experiment-status.json에는 본실험 미실행과 이유를 기록했다. 누락을0초로 주석 처리하지 않았다.','',
      '## 검산','',
      '다운로드5개 파일의ZIP CRC·SHA-256을 확인했다. 원본 임상값·후보 경계·시간 차이를 다시 계산했다. 연구진의 순수 수치 함수를 실행하고 경계를 별도 방식으로 재계산했다. 보수적 비교의450개 최종 모델과5,400개 내부 학습을 별도로 다시 학습해 예측·설정 선택을 재현했다. 이전9개 조건의 예측도 각 집단에서 정확히 재현했다. 같은사람·같은구간·관찰시간은 이전 검증된 입력 및 분할의 바이트/배열 비교로 확인했다.','',
      '마지막 프레임 민감도8조건도 절대 높이 기준으로 별도 재계산해 검산했다. 모델 학습에는 적용하지 않은 진단이다.','',
      '재현: acquire.py → audit.py → study.py → check_inputs.py → train.py → assess.py → verify.py. 추가진단: marker_sensitivity.py → verify_marker_sensitivity.py. 마지막으로 report.py를 실행한다. CPU와 기존 패키지를 사용했고, 원본과 이전 실험은 수정하지 않았다. boundary-comparison.png는 두사례의MOT/TRC 및 자세 변화를 표시한다. 영상 확인을 대신하지 않는다.','']
    (HERE/'STATUS.md').write_text('\n'.join(lines),encoding='utf8')
    save('visual-review.json',{'file':'boundary-comparison.png','scope':'model-derived coordinates only; no actual start signal','ids':['sub-283','sub-288']})
    import platform,sklearn
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'CPU':True,'new_packages':False})
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact-manifest.json'];save('artifact-manifest.json',{p.relative_to(HERE).as_posix():sha(p) for p in files})
    print('REPORT SAVED; strict clinical confirmation unavailable; conservative450models verified',flush=True)
if __name__=='__main__':main()
