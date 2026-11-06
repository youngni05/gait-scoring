"""Write the verified Lite/Full result, preserving the previous Full study."""
import os
from common import *

def main():
    verify=read(HERE/'verification.json');assert verify['passed']
    full=read(FULL/'results.json')['summary'];lite=read(HERE/'results.json')['summary'];op=read(OLD/'results.json')['summary']
    bench=read(HERE/'benchmark.json');pose=read(HERE/'full-lite-pose.json');ext=read(HERE/'extraction.json')
    model=dict(Full=FULL_TASK.stat().st_size,Lite=TASK.stat().st_size)
    changes=[]
    for kind in ['primary','straight']:
        a=full[kind]['best'];b=lite[kind]['best']
        changes.append(dict(kind=kind,Full=a,Lite=b,OpenPose=op[kind]['best'],gap_change=b['normal_minus_abnormal']-a['normal_minus_abnormal'],
            gap_reduction_fraction=1-b['normal_minus_abnormal']/a['normal_minus_abnormal'],normal_shift=b['clip1']['mean']-a['clip1']['mean'],
            abnormal_shift=b['clip0']['mean']-a['clip0']['mean']))
    result=dict(scores=changes,benchmark=bench['summary'],speed_ratio=bench['speed_ratio_full_over_lite'],model_bytes=model,
        model_size_reduction_fraction=1-model['Lite']/model['Full'],pose=pose,
        lite_full_video_loop_seconds=sum(r['loop_seconds'] for r in ext['runs']),verification=verify)
    save(HERE/'comparison.json',result)
    sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'));os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1.1,1.1]})
    colors={'Full':'#b641a8','Lite':'#168db2'}
    for x,variant in enumerate(['Full','Lite']):
        vals=[r['step_seconds'] for r in bench['runs'] if r['variant']==variant];med=bench['summary'][variant]['median_seconds']
        axes[0].scatter(x+np.linspace(-.05,.05,4),vals,c=colors[variant],s=35)
        axes[0].plot([x-.17,x+.17],[med,med],color=colors[variant],lw=3)
        axes[0].text(x,med+.15,f'{med:.2f}초',ha='center',fontsize=11)
    axes[0].set_xticks([0,1],['Full','Lite']);axes[0].set_ylim(0,max(r['step_seconds'] for r in bench['runs'])+.5)
    axes[0].set_xlim(-.5,1.5);axes[0].set_ylabel('50프레임 처리 시간(초)');axes[0].set_title('같은 PC CPU, 번갈아 4회씩')
    for ax,kind,title in zip(axes[1:],['primary','straight'],['기존 입력 13구간','고정 직선 3구간']):
        for x,(variant,r) in enumerate([('Full',full),('Lite',lite)]):
            s=r[kind]['best']
            for offset,clip,label in [(-.17,'clip1','정상'),(.17,'clip0','비정상')]:
                mean=s[clip]['mean'];vals=s[clip]['window_means'];pos=x+offset
                ax.scatter(pos+np.linspace(-.035,.035,len(vals)),vals,color=colors[variant],marker='o' if clip=='clip1' else 's',s=24,alpha=.65)
                ax.plot([pos-.11,pos+.11],[mean,mean],color=colors[variant],lw=3)
                ax.text(pos,mean+1.15,f'{label}\n{mean:.2f}',ha='center',fontsize=9)
            ax.text(x,57,f'차이 +{s["normal_minus_abnormal"]:.2f}',ha='center',color=colors[variant],fontsize=10)
        ax.set_xticks([0,1],['Full','Lite']);ax.set_xlim(-.5,1.5);ax.set_ylim(54,83);ax.set_title(title);ax.set_ylabel('고정 모델 GDI 예측값')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('MediaPipe Lite: 처리 시간은 줄었고, 정상·비정상 점수 차이도 작아졌다',fontsize=15)
    fig.text(.5,.02,'한 사람의 두 영상 탐색. 관절·GDI 정답과 스마트폰 성능은 미검증. 점은 반복 구간 또는 반복 시간 측정이다.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(HERE/'comparison.png',dpi=170);plt.close(fig)
    lines=['# 직접 촬영 영상 실험 — MediaPipe Lite와 Full 비교','',
        '실험 기록상 Lite는 Full보다 모델 파일이 작고, 같은 PC의 반복 속도 비교에서도 빨랐다. 기존 최선 GDI 결합 모델의 정상−비정상 차이는 유지됐으나 Full보다 작아졌다. 사용자가 요청한 기존 Full 결과와 이번 Lite 비교를 각각의 실험으로 보존한다.','',
        '**속도와 모델 크기**','',
        '| 항목 | Full | Lite |','|---|---:|---:|',
        f'| 공식 task 파일 크기 | {model["Full"]/1e6:.2f}MB | {model["Lite"]/1e6:.2f}MB |',
        f'| 동일 50프레임 시간 중앙값 | {bench["summary"]["Full"]["median_seconds"]:.3f}초 | {bench["summary"]["Lite"]["median_seconds"]:.3f}초 |',
        f'| 4회 측정 범위 | {bench["summary"]["Full"]["minimum_seconds"]:.3f}~{bench["summary"]["Full"]["maximum_seconds"]:.3f}초 | {bench["summary"]["Lite"]["minimum_seconds"]:.3f}~{bench["summary"]["Lite"]["maximum_seconds"]:.3f}초 |',
        f'| 중앙값 기준 처리 속도 | {bench["summary"]["Full"]["fps"]:.1f} FPS | {bench["summary"]["Lite"]["fps"]:.1f} FPS |','',
        f'파일 크기는 Lite가 **{100*result["model_size_reduction_fraction"]:.1f}% 작았고**, 이번 CPU 비교 처리 속도는 **{bench["speed_ratio_full_over_lite"]:.2f}배**였다. 파일 크기는 실행 RAM·발열·소비전력 측정값이 아니다.',
        'Intel Core Ultra 7 258V CPU에서 두 영상의 첫 25프레임씩을 미리 읽어 같은 50프레임을 사용했다. Full→Lite, Lite→Full 순서를 번갈아 총 4회씩 실행했다. RGB 변환부터 최종 관절 반환까지 측정했고 영상 읽기·모델 초기화·준비 실행·파일 저장은 제외했다. 영상마다 추적 상태를 초기화하고 CPU·VIDEO·num_poses=1·문턱 각0.5를 동일하게 사용했다.',
        f'Lite가 전체 2,382프레임을 읽고 처리한 별도 실행 시간은 **{result["lite_full_video_loop_seconds"]:.2f}초**였다. 이전 Full 전체 실행은 54.13초였지만 실행 시점과 조건이 같지 않으므로 이 두 수치를 모델 간 속도 배수로 사용하지 않았다. 이전 Full 50프레임 0.87초와 이번 재측정 중앙값 1.79초도 다르다. 이번 Lite 비교에는 같은 시점에 번갈아 측정한 Full 값을 사용하며, 절대 시간 변동의 원인을 별도로 계측하지 않았으므로 특정 원인으로 단정하지 않는다. 스마트폰 실시간 처리 성능은 확인하지 않았다.','',
        '**고정 GDI 모델의 결과**','',
        '| 구간 | 관절 추출 | 정상 | 비정상 | 정상 − 비정상 |','|---|---|---:|---:|---:|']
    for kind,title in [('primary','기존 13구간'),('straight','직선 3구간')]:
        for variant,r in [('OpenPose',op),('Full',full),('Lite',lite)]:
            s=r[kind]['best'];lines.append(f'| {title} | {variant} | {s["clip1"]["mean"]:.2f} | {s["clip0"]["mean"]:.2f} | +{s["normal_minus_abnormal"]:.2f} |')
    lines+=['',
        f'Lite의 차이는 Full보다 기존 구간에서 **{100*changes[0]["gap_reduction_fraction"]:.1f}%**, 직선 구간에서 **{100*changes[1]["gap_reduction_fraction"]:.1f}%** 작았다. 그래도 두 구간 구성 모두 5개 분할 모델에서 정상 점수가 높았다. 분할 모델·구간은 독립 참가자가 아니며, 차이가 작아진 사실을 실제 GDI 정확도 저하로 바꾸어 해석하지 않는다.',
        'Lite의 정상 점수는 Full보다 기존 구간에서 약2.39점, 직선 구간에서 약2.21점 낮았다. 비정상 점수는 각각 약1.02점,0.11점 높았다. 같은 사용자라도 Full과 Lite를 교체하면 실제 움직임이 그대로여도 점수가 달라질 수 있다.',
        '최선 결합은 54개 모델을 그대로 사용했다. 별도 CNN 단독 최종 모델 평균에서는 Lite 기존 구간의 차이가0.47점으로 작았고 5개 분할 중 하나는−1.11점으로 역전됐다. 따라서 작은 단일 CNN으로도 동일하게 유지된 결과라고 설명하지 않는다. 비교 모델 전체 값은 [results.json](results.json)에 보존했다.','',
        '![Full과 Lite 비교](comparison.png)','',
        '**관절 추출 차이**','',
        '| 영상 | Full−Lite 거리 중앙값 | 거리 90백분위 | Full 다리 길이 대비 중앙값 |','|---|---:|---:|---:|']
    for c in pose['clips']:
        q=c['metrics'];name='비정상' if c['clip']=='clip0' else '정상'
        lines.append(f'| {name} | {q["pooled_pixel_distance"]["median"]:.2f}px | {q["pooled_pixel_distance"]["p90"]:.2f}px | {100*q["pooled_leg_length_fraction"]["median"]:.2f}% |')
    lines+=['',
        '원본1080×1920 화면에서 같은 프레임의 양쪽 엉덩이·무릎·발목·발끝, 총8개 지점을 비교했다. Full과 Lite 모두2,382프레임에서 관절을 반환했고, GDI32개 구간이 결측25% 기준을 통과했다.',
        '대표42프레임 검토에서는 같은 사람을 따라갔지만 다리가 겹치는 장면에서 위치 차이가 보였다. 정상 영상의 양쪽 발목 거리90백분위는 약52~54px, 발끝은 약66~69px였다. 이는 Full과 Lite의 불일치이며 어느 추출기가 정답인지는 알 수 없다. 일부 큰 차이가 발생하는 구간을 전체 중앙값만으로 가리지 않는다.',
        'MediaPipe 내부 visibility는 원본 그대로 보존했으며 새 문턱으로 좌표를 지우지 않았다. 신뢰도 수치의 차이를 정확도 우위로 해석하지 않았다. GDI가 요구하는 OpenPose 엄지발가락 대신 MediaPipe 발끝을 근사 대응한다는 제한도 유지했다.',
        '[비정상 Full·Lite 겹침](clip0-full-lite.jpg) · [정상 Full·Lite 겹침](clip1-full-lite.jpg). 자홍색은 Full, 하늘색은 Lite다. [관절별 수치](full-lite-pose.json)에서 비교 함수의 기존 `openpose` 키는 Full 기준, `mediapipe` 키는 Lite 후보를 뜻한다고 명시했다.','',
        '**유지한 조건과 검산**','',
        '원본 영상, 프레임 순서, 실제 시각,4곳의 밀리초 중복 처리,2D픽셀 변환,보간·평활화,32개 분석 구간,모델54개,보정식·결합식을 유지했다. 새 모델 학습·출력 보정·좌우 교환·프레임 재표본화·구간 교체는 하지 않았다. z와 world 좌표는 GDI에 넣지 않았다.',
        '기존 Full 폴더30개 파일의 지문이 모두 유지됐고, 원본 영상2개와 체크포인트54개의 지문을 확인했다. Lite 관절 대응·특징 재계산·Full 대비 거리 중앙값의 차이는0이었다. 예측 결합식을 독립적으로 재계산했고 요약 수치 차이는2.85×10⁻¹⁴점 이하였다. 속도8회와 프레임 수·중앙값도 확인했다. [검산 결과](verification.json)',
        '코드는 [common.py](common.py)의 기존 관절 대응 재사용, [prepare.py](prepare.py)의 기존 전처리 재사용, [predict.py](predict.py)의 고정 모델 호출로 연결된다. 순서는 `fetch_model.py → extract.py → prepare.py → 대표 그림 검토 → predict.py → benchmark.py → verify.py → report.py`다.','',
        '**기록과 적용 범위**','',
        '관측 결과는 Lite의 파일 용량·CPU 처리 시간 이득과 점수 차이 감소를 함께 보여준다. 이 자료만으로 Full·Lite 중 최종 모델을 채택하지 않았다. 여러 사람·다른 촬영·개인 반복 측정, 실제 스마트폰의 속도·RAM·발열·배터리, 실제 관절·GDI 정답에 대한 성능은 미검증이다.',
        '기존 [OpenPose·Full 실험](../20261101-self-video-mediapipe/README.md)을 그대로 보존하고 연구 최상위 README에 연결했다. Lite 모델만 Google 공식 주소에서 다운로드했다. 새 패키지 설치나 개인 영상의 외부 전송은 하지 않았다. 기존 MediaPipe1.2.0·OpenCV5·NumPy·SciPy·PyTorch 환경을 재사용했다.',
        '작은 코드·그림·결과·검산은 이 C 폴더, 다운로드 모델과 관절·특징·예측 배열은 `D:/gait-research-data/D_P/self-video-mediapipe-lite-20261101`에 저장했다.',
        '[전체 비교 수치](comparison.json) · [사전 조건](protocol.json) · [모델 출처·SHA256](model-receipt.json) · [속도 원자료](benchmark.json) · [전체 영상 실행](extraction.json)',
        '',
        '공식 모델 출처: [MediaPipe Pose Landmarker의 Lite·Full 구성](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(dict(speed_ratio=bench['speed_ratio_full_over_lite'],model_bytes=model,gap_reductions=[c['gap_reduction_fraction'] for c in changes]),flush=True)

if __name__=='__main__':main()
