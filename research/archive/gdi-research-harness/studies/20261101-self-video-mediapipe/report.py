"""Summarize verified results and plot the same person's repeated windows."""
import os
from common import *

def main():
    assert read(HERE/'verification.json')['passed']
    a=read(OLD/'results.json')['summary'];b=read(HERE/'results.json')['summary']
    op=read(HERE/'benchmark-openpose.json');mp=read(HERE/'benchmark-mediapipe.json');dec=read(HERE/'benchmark-decode.json')
    ext=read(HERE/'extraction.json');pose=read(HERE/'pose-comparison.json')
    seconds_op=op['step_seconds']+dec['seconds'];seconds_mp=mp['step_seconds']
    pairs=[]
    for kind in ['primary','straight']:
        for model in ['best','CNN_refit','CNN_bag','STT_bag']:
            x=a[kind][model];y=b[kind][model]
            pairs.append(dict(kind=kind,model=model,openpose=x,mediapipe=y,
                normal_shift=y['clip1']['mean']-x['clip1']['mean'],abnormal_shift=y['clip0']['mean']-x['clip0']['mean'],
                gap_change=y['normal_minus_abnormal']-x['normal_minus_abnormal']))
    speed=dict(frames=50,openpose_staged_seconds=seconds_op,mediapipe_seconds=seconds_mp,
        openpose_fps=50/seconds_op,mediapipe_fps=50/seconds_mp,ratio=seconds_op/seconds_mp,
        mediapipe_full_video_frames=sum(r['frames'] for r in ext['runs']),
        mediapipe_full_video_loop_seconds=sum(r['loop_seconds'] for r in ext['runs']),
        scope='CPU same first25 frames per clip, warmup/load/video decode/disk saving excluded; OpenPose separately timed resize+forward+official PAF sum, MediaPipe RGB conversion+final landmarks. Native thread policies differ.')
    save(HERE/'comparison.json',dict(scores=pairs,speed=speed,pose=pose,verification=read(HERE/'verification.json')))
    sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'));os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,kind,title in zip(axes,['primary','straight'],['기존 입력 구간: 영상별 13개','고정한 직선 구간: 영상별 3개']):
        for x,(engine,s,color) in enumerate([('OpenPose',a,'#27805a'),('MediaPipe',b,'#b641a8')]):
            for offset,clip,label in [(-.18,'clip1','정상'),(.18,'clip0','비정상')]:
                vals=s[kind]['best'][clip]['window_means'];pos=x+offset;mean=s[kind]['best'][clip]['mean']
                ax.scatter(pos+np.linspace(-.045,.045,len(vals)),vals,color=color,marker='o' if clip=='clip1' else 's',s=28,alpha=.65)
                ax.plot([pos-.10,pos+.10],[mean,mean],color=color,lw=3)
                ax.text(pos,mean+1.0,f'{label}\n{mean:.2f}',ha='center',fontsize=9)
            ax.text(x,56.5,f'차이 +{s[kind]["best"]["normal_minus_abnormal"]:.2f}',ha='center',fontsize=11,color=color)
        ax.set_xticks([0,1],['OpenPose','MediaPipe Full']);ax.set_xlim(-.5,1.5);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(54,84);axes[0].set_ylabel('동일한 고정 모델의 GDI 예측값')
    fig.suptitle('관절 추출기를 바꿔도 이번 두 영상의 점수 차이는 유지됐다',fontsize=14)
    fig.text(.5,.02,'같은 사람의 반복 구간이다. 실제 GDI·관절 정답 및 스마트폰 성능을 검증한 결과가 아니다.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(HERE/'score-comparison.png',dpi=170);plt.close(fig)
    lines=['# 직접 촬영한 두 영상: MediaPipe Full과 OpenPose 비교','',
        'MediaPipe Full CPU로 정상·비정상 영상 2,382프레임을 처리했다. 기존 GDI 모델과 구간을 유지했을 때 정상 영상의 점수가 더 높은 관계가 유지됐다. 모든 예측값은 실제 GDI 정답과 비교한 정확도가 아니다.','',
        '| 구간 | 관절 추출 | 정상 | 비정상 | 정상 − 비정상 |','|---|---|---:|---:|---:|']
    for kind,title in [('primary','기존 13구간'),('straight','직선 3구간')]:
        for engine,r in [('OpenPose',a),('MediaPipe Full',b)]:
            s=r[kind]['best'];lines.append(f'| {title} | {engine} | {s["clip1"]["mean"]:.2f} | {s["clip0"]["mean"]:.2f} | +{s["normal_minus_abnormal"]:.2f} |')
    lines+=['','두 구간 구성 모두 5개 분할 모델에서 정상 점수가 높았다. 분할 모델과 영상 안의 반복 구간은 독립 참가자가 아니다. MediaPipe의 절대 점수는 두 영상 모두 약 1.2~1.5점 낮았으므로 추출기를 바꾼 전후의 점수를 같은 척도처럼 연결하지 않는다.','',
        '![고정 GDI 모델 점수 비교](score-comparison.png)','',
        '**속도: 동일 PC·동일 50프레임**','',
        f'- OpenPose: 입력 변환·신경망·공식 관절 후처리를 각각 측정해 합산한 시간 **{seconds_op:.3f}초**, 약 **{50/seconds_op:.2f} FPS**.',
        f'- MediaPipe Full: RGB 변환부터 최종 관절 반환까지 **{seconds_mp:.3f}초**, 약 **{50/seconds_mp:.2f} FPS**.',
        f'- 이번 구현·설정의 단계별 비교에서 MediaPipe는 약 **{seconds_op/seconds_mp:.1f}배** 빨랐다. 파일 읽기·모델 초기화·준비 실행·결과 저장은 제외했다.',
        f'- MediaPipe가 두 영상 전체를 읽고 처리한 반복문 시간은 **{speed["mediapipe_full_video_loop_seconds"]:.2f}초**였다. 프레임 건너뛰기 없이 총 2,382프레임을 처리했다. 이는 별도의 전체 영상 실행 기록이며 50프레임 속도와 측정 범위가 다르다.',
        '- CPU는 Intel Core Ultra 7 258V다. OpenPose는 PyTorch CPU 4스레드, 영상 처리 2스레드, MediaPipe는 CPU XNNPACK 기본 스레드 설정이다. 두 엔진을 동시에 실행하지 않았다. 현재 구현의 비교이며 모든 최적화된 OpenPose 구현이나 스마트폰의 속도로 일반화하지 않는다.',
        '- 기존 학교 GPU OpenPose의 132.32초 기록은 저장까지 포함한 다른 하드웨어·측정 범위이므로 속도 배수 계산에 사용하지 않았다.','',
        '**관절 추출 차이**','',
        '| 영상 | 대응 8지점 거리 중앙값 | 거리의 90백분위 | 다리 길이 대비 중앙값 |','|---|---:|---:|---:|']
    for c in pose['clips']:
        title='비정상' if c['clip']=='clip0' else '정상'
        lines.append(f'| {title} | {c["pooled_pixel_distance"]["median"]:.2f}px | {c["pooled_pixel_distance"]["p90"]:.2f}px | {100*c["pooled_leg_length_fraction"]["median"]:.2f}% |')
    lines+=['',
        '거리는 원본 1080×1920 화면 좌표에서 계산했다. 다리 길이는 OpenPose의 양쪽 엉덩이–무릎–발목 연결 길이 평균이다. 이는 두 방법의 불일치이며 어느 쪽이 정답인지 알려주는 오차가 아니다.',
        'MediaPipe는 전 프레임에서 관절을 반환했다. OpenPose는 비정상 영상의 20프레임에서 여러 사람이 검출돼 기존 규칙에 따라 결측으로 남겼다. MediaPipe는 한 사람만 찾는 설정이어서 이 차이를 검출 정확도 우위로 해석하지 않는다. 관절 반환이 위치의 정확성을 뜻하지도 않는다.',
        'MediaPipe의 왼쪽 무릎 visibility가 0.5 미만인 프레임은 비정상 136개, 정상 55개였다. 신뢰도 체계가 다른 OpenPose와 수치를 직접 비교하지 않았다. 기존 전처리가 신뢰도 열을 제거하므로 이번에도 새 신뢰도 문턱을 적용하지 않았고 원래 좌표를 보존했다.',
        '점수 확인 전에 기존 직선 구간의 대표 42프레임을 겹쳐 검토했다. 두 추출기는 같은 사람의 다리를 대체로 따라가지만 가림·겹침에서 무릎·발목·발끝 차이가 보인다. 모든 프레임이나 해부학적 좌우의 정확성을 보증한 검토는 아니다.',
        '[비정상 영상 겹침 그림](clip0-comparison.jpg) · [정상 영상 겹침 그림](clip1-comparison.jpg). 초록색은 OpenPose, 자홍색은 MediaPipe다.','',
        '**입력과 모델을 어떻게 유지했는가**','',
        'MediaPipe의 2D 화면 좌표를 원본 픽셀로 바꾼 뒤, 양쪽 엉덩이·무릎·발목과 발끝을 기존 전처리에 넣었다. 발끝(foot_index)은 OpenPose의 엄지발가락과 완전히 같은 지점이 아니며 근사 대응이다. 3D world 좌표와 깊이 z는 사용하지 않았다.',
        '원본 전체 기록의 보간·평활화, 124프레임 구간 32개, 결측 25% 제한, 모델 54개, 보정식과 결합 비중을 유지했다. 32개 구간 모두 통과했다. 모델 학습·보정 변경·영상 반전·좌우 교환·점수에 따른 구간 교체는 하지 않았다.',
        'MediaPipe는 CPU·VIDEO 모드·num_poses=1·검출/존재/추적 문턱 각 0.5이며 매 영상에서 추적 상태를 초기화했다. 정상 영상의 매우 가까운 원본 시각 4곳은 밀리초 반올림 시 중복되어 추적기에만 이전 시각+1ms를 전달했다. 원본 프레임과 GDI 구간의 순서·길이는 바꾸지 않았다.','',
        '**문제 발견과 검산**','',
        '첫 CPU 속도 재실행에서 OpenCV 5의 크기 조정 결과가 기존 공식 OpenPose 입력과 달랐다. 그 결과와 측정 기록은 진단 자료로 보존하고, 공식 기준 입력 3개와 차이 0으로 일치하는 OpenCV 4.8.1 변환으로 다시 측정했다. 두 버전이 원본 2,382프레임을 디코딩한 픽셀 자체는 모두 SHA256이 일치했다. 이 수정은 속도 측정의 OpenPose 입력 경로에만 적용하며 기존 GDI 비교의 OpenPose 좌표는 처음부터 원래 저장값을 사용했다.',
        '공식 후처리 준비 실행 중에는 임시 배열 수명 때문에 네이티브 모듈이 종료되는 문제가 있었다. 배열을 처리 종료까지 유지하도록 수정했고 이후 정상 완료했다. 입력 기준이나 허용 오차는 완화하지 않았다.',
        '원본 영상 2개·MediaPipe 출력·모델 54개의 SHA256, 관절 대응, 특징 재계산, 독립 결합식·평균 계산을 확인했다. 기존 OpenPose 특징도 차이 0으로 재현했다. 자세한 검산 수치는 [verification.json](verification.json)에 있다.','',
        '**활용 범위와 재현 위치**','',
        '이번 영상 쌍에서는 MediaPipe Full이 빠르고 기존 GDI의 정상−비정상 차이도 유지돼, 모바일 관절 추출의 후속 후보로 검토할 근거가 생겼다. 다만 한 사람의 두 조건이며 실제 관절·GDI 정답, 여러 사람의 재현성, 경미한 개인 변화, 스마트폰의 속도·발열·배터리는 확인하지 않았다. Lite·Heavy 변형은 실행하지 않았다.',
        '핵심 코드는 [common.py](common.py)의 관절 대응과 [predict.py](predict.py)의 기존 예측기 호출이다. `extract.py → prepare_compare.py → 대표 그림 검토 → predict.py` 순서로 처리했다. 속도는 `prepare_benchmark_inputs.py(Python 3.7) → benchmark.py openpose → benchmark.py mediapipe → benchmark_decode.py(Python 3.7)`로 확인하며, 마지막에 `verify.py → report.py`를 실행한다.',
        'MediaPipe 1.2.0과 기존 Full 모델·OpenCV·NumPy·SciPy·PyTorch를 재사용했다. 새 설치·다운로드·서버 전송 없이 로컬에서 처리했다. 작은 결과는 이 C 폴더, 관절·특징·예측·측정 배열은 `D:/gait-research-data/D_P/self-video-mediapipe-20261101`에 있다.',
        '[전체 비교 수치](comparison.json) · [사전 조건](protocol.json) · [추출 기록](extraction.json) · [관절별 차이](pose-comparison.json) · [원본 예측 결과](results.json)',
        '',
        '구현 근거: [MediaPipe 공식 Python 영상 처리 안내](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python), [33개 관절과 모델 구성](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(speed,indent=2),flush=True)

if __name__=='__main__':main()
