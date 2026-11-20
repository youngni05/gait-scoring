"""App wording uses only the original report's displayable source observations."""
from history import compare_history

PRIORITY={'cadence':0,'supported_knee_range':1,'step_time':2}
LABELS={'cadence':'걸음 빈도','supported_knee_range':'무릎 움직임 폭','step_time':'걸음 시간'}
DECIMALS={'cadence':0,'supported_knee_range':1,'step_time':2}
SIDE={'left':'왼쪽','right':'오른쪽','bilateral':'양쪽','none':''}


def display_gate(request,row,aggregate):
    try:return _display_gate(request,row,aggregate)
    except (KeyError,TypeError,AttributeError):return False


def _display_gate(request,row,aggregate):
    """Every source/segment contributing to this exact aggregate must be displayed."""
    visits=[request['current_visit'],*request['prior_visits']]
    used=aggregate.get('used_records',[])
    if not used:return False
    metric=row['metric_id'];side=row['side']
    labels={'cadence':('rhythm','분당 걸음 수'),
            'supported_knee_range':('knee',f'{SIDE[side]} 무릎 움직임 폭'),
            'step_time':('step_time',f'{SIDE[side]} 발로 내딛는 시간')}
    if metric not in labels:return False
    item_id,label=labels[metric]
    for source in used:
        matches=[record for visit in visits if visit['visit_id']==aggregate['visit_id']
                 for record in visit['records'] if record['analysis_id']==source['analysis_id']
                 and record['source_id']==source['source_id'] and record['source_sha256']==source['source_sha256']]
        if not matches or not source.get('segment_ids'):return False
        for record in matches:
            analysis=record['analysis'];report=analysis.get('current_report',{})
            if report.get('schema')!='ai-current-report/1' or report.get('source_id')!=record['source_id']:return False
            for segment_id in source['segment_ids']:
                original=[s for s in analysis['segments'] if s['segment_id']==segment_id and s['direction']==row['direction']]
                shown=[s for s in report.get('segments',[]) if s['segment_id']==segment_id and s['direction']==row['direction']]
                if len(original)!=1 or len(shown)!=1:return False
                raw=original[0]['knee' if metric=='supported_knee_range' else 'gait']['metrics'][metric]
                raw=raw[side] if metric!='cadence' else raw
                items=[m for m in shown[0]['metrics'] if m['id']==item_id and m['status'] in ('observed','limited')]
                if not any(v['label']==label and v['unit']==row['unit'] and v['value']==raw['value']
                           for m in items for v in m['display_values']):return False
    return True


def recent_observations(row):
    ids=set(row['recent']['reference_visit_ids'])
    return [v for v in row['timeline'] if not v['is_current'] and v['visit_id'] in ids]


def explain(comparison,request):
    candidates=[];withheld=[]
    for row in comparison['metrics']:
        if row['metric_id'] not in PRIORITY or row['recent']['delta'] is None or row['direction']=='unknown':continue
        recent=recent_observations(row)
        if not recent or not all(display_gate(request,row,v) for v in [row['current'],*recent]):
            withheld.append(dict(metric_id=row['metric_id'],side=row['side'],direction=row['direction'],
                                 condition_key_id=row['condition_key_id'],reason='contributing_observation_not_displayable'))
            continue
        candidates.append(row)
    candidates.sort(key=lambda r:(PRIORITY[r['metric_id']],r['direction'],r['side'],r['condition_key_id']))
    selected=[];families=set()
    for row in candidates:
        if row['metric_id'] in families:continue
        families.add(row['metric_id']);selected.append(row)
        if len(selected)==2:break
    for row in candidates:
        if len(selected)==2:break
        if row not in selected:selected.append(row)
    evidence=[]
    for r in selected:
        metric=r['metric_id'];d=r['recent']['delta'];digits=DECIMALS[metric]
        previous=f"{r['recent']['value']:.{digits}f}";current=f"{r['current']['value']:.{digits}f}"
        shown_delta=round(float(current)-float(previous),digits)
        name=LABELS[metric];side=SIDE[r['side']]
        direction={'image_left':'화면 왼쪽으로 걷는 구간','image_right':'화면 오른쪽으로 걷는 구간'}[r['direction']]
        if previous==current:meaning=f'{side} {name}은 표시 정밀도에서 이전값과 이번값이 같다'
        elif metric=='supported_knee_range' and abs(d)<1:meaning=f'{side} 무릎 움직임 폭의 관측값 차이는 1도 미만이다'
        elif metric=='cadence':meaning='같은 시간에 디딘 걸음이 '+('많아졌다' if d>0 else '적어졌다')
        else:meaning=f'{side} {name}의 관측값이 '+('늘었다' if d>0 else '줄었다')
        distribution=r['past_distribution'];n=distribution['n_dates'];lo,hi=distribution['q1'],distribution['q3']
        past=[v for v in r['timeline'] if not v['is_current']]
        distribution_allowed=bool(past) and all(display_gate(request,r,v) for v in past)
        in_range=distribution_allowed and n>=3 and lo is not None and hi is not None and lo<=r['current']['value']<=hi
        evidence.append(dict(metric_id=metric,label=name,side=r['side'],direction=r['direction'],direction_label=direction,
            condition_key_id=r['condition_key_id'],unit=r['unit'],previous=r['recent']['value'],current=r['current']['value'],delta=d,
            display_previous=previous,display_current=current,display_delta=shown_delta,display_decimals=digits,
            reference_visit_ids=r['recent']['reference_visit_ids'],reference_captured_at=r['recent']['captured_at'],
            meaning=meaning.strip(),in_prior_iqr=in_range,prior_date_count=n,prior_distribution_display_eligible=distribution_allowed))
    if not evidence:
        state='first_visit' if not request['prior_visits'] else 'conditions_unavailable'
        sentence=('이 대상자의 이전 기록이 없어 이번 관측부터 남긴다.' if state=='first_visit' else
                  '원시값은 비교할 수 있으나 두 기록에서 표시 가능한 관측 근거가 부족해 차이 설명을 보류한다.' if withheld else
                  '같은 조건의 이전값을 연결하지 못해 이번과의 차이는 보류한다.')
    elif all(e['display_delta']==0 for e in evidence):state='unchanged';sentence='표시 정밀도에서 비교한 항목의 이전값과 이번값이 같다.'
    elif all(e['in_prior_iqr'] for e in evidence):state='within_prior_range';sentence='비교한 이번값은 과거 여러 날짜에서 관측된 가운데 범위 안에 있다.'
    elif all(e['metric_id']=='supported_knee_range' and abs(e['delta'])<1 for e in evidence):
        state='changes'
        if len(evidence)==2 and {e['side'] for e in evidence}=={'left','right'}:
            detail='왼쪽과 오른쪽 모두 측정값 차이가 1도 미만입니다.'
        else:
            descriptions=[f"{e['direction_label']}의 {SIDE[e['side']]} 무릎" for e in evidence]
            detail=' · '.join(descriptions)+' 측정값 차이가 '+('각각 ' if len(evidence)>1 else '')+'1도 미만입니다.'
        sentence='두 기록에서 무릎을 굽혔다 펴는 폭이 비슷하게 보였습니다. '+detail
    elif any(e['display_delta']>0 for e in evidence) and any(e['display_delta']<0 for e in evidence):
        state='mixed_changes';sentence='항목별로 늘어난 값과 줄어든 값이 함께 관측됐다.'
    else:
        state='changes';sentence=next(e['meaning'] for e in evidence if e['display_delta']!=0)+' — 같은 방향·방법의 최근 관측과 비교했다.'
    return dict(schema='gait-history-explanation/1',state=state,sentence=sentence,evidence=evidence,withheld=withheld,
        limits='관측값 차이이며 보행 속도·균형·악화 판정이 아니다. 표시 반올림은 측정 오차 기준이 아니며, 과거 가운데 범위도 검증된 반복측정 오차 기준은 아니다.')


def compare_and_explain(request):
    comparison=compare_history(request)
    return dict(comparison=comparison,explanation=explain(comparison,request))
