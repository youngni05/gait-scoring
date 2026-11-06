"""Korean E1 tables and the existing storyline's slide8 section."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as base
import pandas as pd

TITLES = {'mobilise_sppb': 'Mobilise-D SPPB·주간 센서', 'mobilise_tug': 'Mobilise-D TUG·주간 센서',
          'opencap_tug': 'OpenCap TUG·3초', 'opencap_patterns_tug': 'OpenCap TUG·6초 주기 후보',
          'older44_sppb': '고령자 44명 SPPB·Vicon', 'mobilise_nonwalk': 'Mobilise-D 균형＋일어서기·주간 센서'}
GROUPS = {'speed': '속도', 'stride_length': '보폭', 'cadence': '걸음 빈도', 'cycle_duration': '보행 주기',
          'pelvis_angles': '골반 각도', 'hip_angles': '고관절 각도', 'knee_angles': '무릎 각도',
          'ankle_angles': '발목·거골하 관절각', 'lumbar_angles': '몸통 각도',
          'cycle_rhythm': '주기·변동·좌우 차이', 'leg_ROM_asymmetry': '다리 관절 범위·좌우 차이',
          'hip_knee_coordination': '고관절·무릎 협응', 'pelvis_lumbar_ROM': '골반·몸통 범위',
          'step_stride_time': '걸음·보행 주기 시간', 'gait_events': '보행 사건 시점',
          'support_time': '한발·두발 지지시간', 'step_geometry': '걸음길이·보폭·너비', 'limp_index': '제공처 Limp Index'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---']*len(headers)) + '|']
                     + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def interval(delta, low, high):
    def fmt(value):
        return f'{value:+.6f}' if value != 0 and round(value, 3) == 0 else f'{value:+.3f}'
    return f'{fmt(delta)} [{fmt(low)}, {fmt(high)}]'


def main():
    p = read(HERE / 'protocol.json')
    names = p['primary_datasets'] + p['supporting_datasets']
    summaries, groups, checks = {}, {}, {}
    for name in names:
        summaries[name] = read(HERE / f'{name}-summary.json')
        groups[name] = pd.read_csv(HERE / f'{name}-groups.csv')
        checks[name] = read(HERE / f'{name}-verification.json')
        assert checks[name]['passed']
        assert summaries[name]['group_table_sha256'] == base.sha(HERE / f'{name}-groups.csv')
    pd.concat(groups.values(), ignore_index=True).to_csv(HERE / 'all-groups.csv', index=False, encoding='utf-8-sig')
    total_rows = []
    slide_rows = []
    for name in names:
        s = summaries[name]
        ci = s['full_minus_speed']
        unit = '초' if name.endswith('tug') else '점'
        row = [f'{TITLES[name]} ({s["people"]}명)', f'{s["speed_MAE"]:.3f}', f'{s["full_MAE"]:.3f}',
               interval(ci['delta'], ci['ci_low'], ci['ci_high'])]
        total_rows.append(row + [unit])
        if name in p['primary_datasets']:
            slide_rows.append(row)
    evidence = []
    for name in p['primary_datasets']:
        found = summaries[name]['additional_families']
        evidence.append(f'- **{TITLES[name]}:** ' + ('속도에 ' + '·'.join(GROUPS[k] for k in found) + ' 묶음을 더한 비교가 사전 기록한 추가 정보 기준을 충족했다.'
                       if found else '이번 비교에서 추가 정보 기준을 충족한 지표 묶음은 없었다.'))
    shared_removal = [k for k in ['stride_length', 'cadence', 'cycle_duration']
                      if all(bool(groups[name].set_index('family').loc[k, 'removal_support'])
                             for name in ['mobilise_sppb', 'mobilise_tug'])]
    if shared_removal:
        evidence.append('- **겹치는 정보 점검:** ' + '·'.join(GROUPS[k] for k in shared_removal)
                        + '는 두 주간 센서 정답 모두에서 전체 입력에서 빼도 오차가 증가했다. 다만 TUG의 걸음 빈도 제거 효과는 95% 범위 하한이 0에 가깝다. 지표 묶음의 효과이며 그 안의 개별 값이나 원인을 분리한 결과는 아니다.')
    slide = '''### 8. 속도 외에 어떤 보행 정보가 더 필요한가 (RQ3)

**핵심:** 주간 센서에서는 속도 외 지표가 추가 정보를 주었고, OpenCap 3초에서는 추가 이득을 확인하지 못했다.

**비교:** 같은 사람·분할·RF에서 속도 요약 묶음에 지표를 하나씩 추가하고, 전체 입력에서는 하나씩 제거했다.

''' + table(['자료·대상', '속도만 MAE', '전체 지표 MAE', '전체−속도 차이 [95% 범위]'], slide_rows) + '''

음수는 오차 감소다. Mobilise-D SPPB는 점, TUG는 초 단위이며 행 사이 성능 순위를 매기지 않는다.

**발표 하단 주석:** 개발자료 후속 탐색 · 사람 단위 평가 · 뒤섞기 대조 · 10,000회 재표집 · 다중비교 보정 전

#### 발표자 메모 (슬라이드 본문과 구분)

E1 보완: 2026-11-01. 표 대신 아래 그림을 사용할 수 있다.

''' + '\n'.join(evidence) + '''

**해석:** 주간 센서 집계값에서 확인한 추가 정보를 짧은 영상에서 바로 측정할 수 있다고 볼 수는 없다. 다음 측정 실험에서는 속도를 중심으로, 추가 정보가 나타난 지표를 실제로 안정되게 잴 수 있는지 확인한다.

**별도 회전 근거:** 이전 238명 관찰시간 맞춤 OpenCap 실험에서는 직선 보행 2.114초 → 회전 추가 1.118초였다. 회전은 예측 대상 TUG 과제의 별도 기록에서 가져왔으며, 임상 계시 시행과 파일의 정확한 대응은 미확정이다. 이번 E1과 합치거나 평소 직선 걸음만의 성능으로 설명하지 않는다.

**발표 주석:** 사람 5겹×3회 반복, 5개 난수 뒤섞기 대조, 사람 단위 10,000회 짝지은 재표집. 기존 결과를 본 뒤 보완한 개발자료 탐색이며 95% 범위는 학습된 모델에 조건부이고 다중비교 보정 전이다. 개선 미확인은 정보가 없다는 증명이 아니다.

**전환 문장:** 점수 추정 연구에서 확인한 정보원을 바탕으로, 집 안에서 무엇을 정확히 측정할지 좁혀 간다.

**그림·상세표:** [슬라이드용 그림](../gdi-research-harness/studies/20261031-gait-measurement-discovery/e1-supplement/slide8-e1.png) · [E1 결과와 지표 정의](../gdi-research-harness/studies/20261031-gait-measurement-discovery/e1-supplement/RESULTS.md)
'''
    standalone_slide = slide.replace('../gdi-research-harness/studies/20261031-gait-measurement-discovery/e1-supplement/', '')
    (HERE / 'slide8.md').write_text(standalone_slide, encoding='utf-8')
    root = HERE.parents[3]
    storyline_path = root / 'docs/interim-storyline-v2.md'
    storyline = storyline_path.read_text(encoding='utf-8')
    start = storyline.index('### 8.')
    end = storyline.index('\n---\n\n## 3막.', start)
    updated = storyline[:start] + slide.rstrip() + '\n' + storyline[end:]
    # Change only the section which the user authorized.
    storyline_path.write_text(updated, encoding='utf-8')
    detail = '''# E1 보완 결과 — 속도 외 지표의 추가 정보

기존 대상·입력·사람 분할·RF 설정을 유지하고, 속도 대비 지표 묶음 추가·전체에서 묶음 제거·사람 대응 뒤섞기 대조를 비교했다. 원래 결과를 본 뒤 진행한 후속 탐색이며, 독립 검증으로 해석하지 않는다.

## 발표에 사용할 결과

E1 보완: 2026-11-01.

''' + '\n'.join(evidence) + '\n\n' + table(['자료·대상', '속도만 MAE', '전체 지표 MAE', '전체−속도 차이 [95% 범위]', '단위'], total_rows) + '''

MAE는 사람별 실제 점수·시간과 예측의 차이를 절댓값으로 평균한 값이다. 같은 사람의 3회 평가 오차를 먼저 평균한다. 반복 예측을 먼저 평균한 앙상블 성능이 아니다. 사람 수와 입력이 달라 자료 사이 숫자로 우열을 정하지 않는다.

## 지표 묶음별 추가·제거 결과

추가 차이는 **속도＋묶음 − 속도**이므로 음수가 개선이다. 뒤섞기 차이는 **실제 속도＋묶음 − 속도＋뒤섞은 묶음**이므로 음수가 실제 사람 대응의 이점이다. 제거 차이는 **전체−묶음의 오차 − 전체 오차**이므로 양수는 해당 묶음을 빼면 나빠졌다는 뜻이다. 지표끼리 정보가 겹칠 수 있어, 추가 이득과 제거 손실이 반드시 동시에 나타나지는 않는다.

추가 정보 기준은 추가 차이의 95% 범위가 0 아래, 3회 중 2회 이상 개선, 뒤섞기 차이의 95% 범위가 0 아래, 뒤섞기 5개 난수 중 4개 이상에서 실제 입력이 우세한 경우다. 이는 탐색 결과의 표현 기준이며 확인적 유의성 판정이 아니다. 모든 예정 묶음을 아래에 제시한다.

0에 가까운 수치는 소수 여섯 자리까지 표시해 구간이 0을 포함하는지 반올림으로 혼동하지 않도록 했다.
'''
    for name in names:
        rows = []
        for _, r in groups[name].iterrows():
            remove = interval(r.removal_delta, r.removal_ci_low, r.removal_ci_high)
            if r.family == 'speed':
                rows.append([GROUPS[r.family], '—', '—', '—', '제거만 비교', remove])
            else:
                rows.append([GROUPS[r.family], interval(r.addition_delta, r.addition_ci_low, r.addition_ci_high),
                    f'{int(r.addition_negative_repeats)}/3', interval(r.real_minus_shuffle, r.shuffle_ci_low, r.shuffle_ci_high),
                    '추가 정보 근거' if bool(r.additional_information) else '추가 정보 미확인', remove])
        detail += '\n## ' + TITLES[name] + '\n\n' + table(['묶음', '추가 차이 [95%]', '개선 반복', '실제−뒤섞기 [95%]', '판단', '제거 차이 [95%]'], rows) + '\n'
    detail += '''
## 무엇을 입력하고 무엇을 맞혔는가

- **Mobilise-D:** 주간 센서 요약에서 실제 SPPB 총점·TUG 초를 각각 예측했다. SPPB 2,232명·TUG 1,980명이며, 이전 긴 보행 구간 실험의 2,174명·1,939명과 다른 집단이다. 속도 5개, 보폭 3개, 걸음 빈도 4개, 보행 주기 3개의 기존 묶음을 유지했다. 변동계수는 원래의 속도·보폭·걸음 빈도·주기 묶음 안에 포함되어 있으므로, 이번 묶음 결과만으로 변동성 하나의 독립 효과를 말하지 않는다. 균형＋일어서기 2,231명은 보행 구성점수를 제외한 기존 보조 정답이다.
- **OpenCap 3초:** 298명의 여러 카메라 복원 시계열에서 실제 TUG를 예측했다. 속도 기준은 수평·수직 속도의 평균·표준편차·인접 변화량 6개다. 나머지는 골반·고관절·무릎·발목·몸통의 기존 각도 요약 묶음이다. 기존 속도 안정성 실험의 2.058초와 사람 분할·RF 설정이 달라 같은 수치의 재현으로 설명하지 않는다.
- **OpenCap 6초:** 195명의 기존 직선 보행과 무릎 굽힘 봉우리 기반 주기 대리값을 사용했다. 속도 기준은 수평 속도 요약 3개다. 발 접촉 정답으로 검증한 보행 주기는 아니다. 기존 제외 대상 sub-283을 그대로 제외했고, 298명과의 차이를 보행 길이 효과로 해석하지 않는다.
- **고령자 44명:** 제공처 Vicon 보행 지표에서 좌우 평균·절대 차이를 구하고 5회 보행을 평균했다. 3초로 자른 입력이 아니라 전체 보행 지표이며 실제 SPPB 범위는 5~12점이다. 표본이 작아 보조 분석으로 둔다.
- 이번 고정 입력에는 회전·의자·실제 5회 일어서기 정답 비교가 없다. 이를 새로 추출하거나 회전 근거와 합치지 않았다. 단일 카메라 RGB 측정 정확도는 이 분석으로 확인하지 못한다.

## 실행과 검산

사람 분할은 기존 난수 17·41·89의 5겹 평가를 그대로 썼다. RF는 나무 128개, 전체 특징 사용, 최소 잎 인원은 500명 이상 자료에서 10명·그 외 2명이다. 보완 과정에서는 설정·지표·대상을 다시 선택하지 않았다. 기존 고정 후보 예측을 재사용했고 새 학습은 뒤섞기 대조와 검산을 위한 재학습이다.

각 지표 묶음의 열들을 함께 묶어 학습 사람 안과 평가 사람 안에서 각각 섞었다. 속도·정답·분할은 유지했다. 5개 난수는 추가 특징 수에 대한 대조이며 정확한 순열 검정이 아니다. 뒤섞기는 속도와 해당 지표의 관계도 바꾸므로 인과관계나 조건부 독립성을 입증하지 않는다.

95% 범위는 사람 단위 짝지은 10,000회 재표집이다. 같은 사람의 반복 오차를 먼저 평균해 반복을 독립 사람처럼 세지 않았다. 전체 재학습·새 기관·새 집단 변동을 포함하지 않으며, 여러 묶음·정답의 비교에 대한 보정도 하지 않았다. 기존 분석을 재사용하는 탐색 결과이므로 추가 정보 미확인을 동등성이나 무가치함의 증거로 쓰지 않는다.

실행 흐름은 `run_e1.py freeze → run → analyze → verify_e1.py → report_e1.py`다. 핵심은 `shuffled_input`에서 학습·평가 사람의 경계를 유지하며 지표 대응만 끊고, `analyze`에서 실제·속도 기준·뒤섞기의 사람별 오차를 같은 순서로 비교하는 부분이다. 기존 NumPy·pandas·scikit-learn·Matplotlib를 재사용했으며 새 패키지나 GPU는 사용하지 않았다.

모든 대조 예측의 사람·정답·분할·누락·중복과 저장 지문, 지표 묶음의 분할 내 순열을 확인했다. 자료마다 기존 속도·전체·추가 후보 3개 모델과 뒤섞기 1개 모델을 독립 경로로 재학습하고, 저장한 대표 모델을 다시 불러 예측을 대조했다. 모든 대조 모델을 다시 학습한 것은 아니다. 모든 표의 MAE·반복 방향·10,000회 구간·판정은 저장 예측에서 별도로 재계산했다. 구체적인 수와 최대 예측 차이는 자료별 `*-verification.json`에 있다.

작은 코드·표·그림·검산 JSON은 이 C 폴더에, 대조 예측·순열 기록·대표 모델은 `D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gait-measurement-discovery/e1-supplement`에 저장했다. 기존 원자료·예측·프로토콜은 변경하지 않았다. 실행 조건은 [protocol.json](protocol.json), 모든 묶음의 숫자와 난수별 대조는 [all-groups.csv](all-groups.csv), 발표 문안은 [slide8.md](slide8.md)에서 확인할 수 있다.
'''
    (HERE / 'RESULTS.md').write_text(detail, encoding='utf-8')
    os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
    sys.path.insert(0, str(root / 'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams['font.family'] = font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.73, wspace=.63)
    for ax, name in zip(axes, p['primary_datasets']):
        frame = groups[name].query('family != "speed"').reset_index(drop=True)
        for i, r in frame.iterrows():
            color = '#087d65' if bool(r.additional_information) else '#677487'
            ax.plot([r.addition_ci_low, r.addition_ci_high], [i, i], color=color, linewidth=2)
            ax.scatter(r.addition_delta, i, s=45, color=color, zorder=3)
        ax.axvline(0, color='#1f2937', linewidth=1, linestyle='--')
        ax.set_yticks(range(len(frame)), [GROUPS[k] for k in frame.family], fontsize=10)
        ax.set_ylim(len(frame)-.5, -.5)
        ax.set_title(TITLES[name] + f'\n{summaries[name]["people"]:,}명', fontsize=13, pad=14)
        ax.set_xlabel('오차 변화 (' + ('초' if name.endswith('tug') else '점') + ')', fontsize=11)
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('속도에 지표 묶음을 더했을 때의 예측 오차 변화', fontsize=20, y=.96, fontweight='bold')
    fig.text(.5, .84, '점: 평균 차이  |  선: 사람 단위 10,000회 재표집 95% 범위  |  0 왼쪽: 개선', ha='center', fontsize=12)
    fig.text(.5, .07, '초록: 뒤섞기 대조·반복 방향까지 추가 정보 기준 충족  ·  회색: 추가 정보 미확인\n개발자료 후속 탐색 · 모델에 조건부인 범위 · 다중비교 보정 전 · 자료별 단위·측정 조건이 다름', ha='center', fontsize=10, color='#465366')
    fig.savefig(HERE / 'slide8-e1.png', dpi=180, facecolor='white')
    fig.savefig(HERE / 'slide8-e1.svg', facecolor='white')
    plt.close(fig)
    write_paths = [HERE / f for f in ['RESULTS.md', 'slide8.md', 'slide8-e1.png', 'slide8-e1.svg', 'all-groups.csv']]
    write_paths.append(storyline_path)
    base.save(HERE / 'report-manifest.json', {'sha256': {str(path): base.sha(path) for path in write_paths},
              'turn_reference_source_sha256': base.sha(HERE.parent.parent / '20261031-opencap-turn/STATUS.md'),
              'report_code_sha256': base.sha(Path(__file__)), 'storyline_only_slide8_replaced':
              updated[:start] == storyline[:start] and updated[updated.index('\n---\n\n## 3막.', start):] == storyline[end:]})
    print('REPORT_WRITTEN', str(storyline_path), flush=True)


if __name__ == '__main__':
    main()
