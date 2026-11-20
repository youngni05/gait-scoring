import { reportMetric, visitKneeSummary } from './report-values.mjs';

const directions = ['image_left', 'image_right'];
const sides = ['left', 'right'];
const mean = a => a.reduce((sum, v) => sum + v, 0) / a.length;

export function walkingEstimate(visit, direction) {
  const intervals = new Map();
  for (const segment of visit.analysis.segments) {
    if (!directions.includes(segment.direction) || (direction && direction !== segment.direction)) continue;
    const gait = segment.gait;
    const tracking = reportMetric(segment, visit.analysis.current_report, 'rhythm')?.evidence?.tracking;
    if (!tracking || !Array.isArray(tracking.transitions)) continue;
    let steps = visit.local_step_intervals_by_segment?.[segment.segment_id];
    if (!Array.isArray(steps)) {
      if (tracking.status === 'unavailable' || tracking.checked_transitions !== tracking.possible_transitions ||
          !Array.isArray(gait?.quality_reasons) || gait.quality_reasons.length) continue;
      steps = gait.steps ?? [];
    }
    for (const { start_ms, stop_ms, landing_side } of steps) {
      const seconds = (stop_ms - start_ms) / 1000;
      if (!Number.isFinite(start_ms) || !Number.isFinite(stop_ms) || !sides.includes(landing_side) ||
          start_ms < segment.start_ms || stop_ms > segment.stop_ms || seconds < .2 || seconds > 1.5) continue;
      if (tracking.transitions.some(t => t.source_time_ms >= start_ms - tracking.contact_radius_ms &&
          t.source_time_ms <= stop_ms + tracking.contact_radius_ms)) continue;
      intervals.set(start_ms + '/' + stop_ms, { segmentId:segment.segment_id, direction:segment.direction,
        start_ms, stop_ms, side:landing_side, seconds });
    }
  }
  const used = [...intervals.values()];
  const time = side => {
    const values = used.filter(s => s.side === side).map(s => s.seconds);
    return { value:values.length ? mean(values) : null, count:values.length };
  };
  return { cadence:used.length >= 2 ? 60 / mean(used.map(s => s.seconds)) : null,
    left:time('left'), right:time('right'), count:used.length, intervals:used };
}

function directionMetrics(visit, direction) {
  const segments = visit.analysis.segments.filter(s => s.direction === direction);
  const timing = walkingEstimate(visit, direction);
  const rows = [
    ['cadence', '분당 걸음 수', timing.cadence, '걸음/분', timing.count],
    ['left-time', '왼발 걸음 시간', timing.left.value, '초', timing.left.count],
    ['right-time', '오른발 걸음 시간', timing.right.value, '초', timing.right.count],
  ];
  const knees = visitKneeSummary({...visit, analysis:{...visit.analysis, segments}});
  for (const row of knees) rows.push([row.side+'-knee', (row.side === 'left' ? '왼쪽' : '오른쪽')+' 무릎 움직임 폭',
    row.value, '°', row.frames]);
  return rows.filter(row => Number.isFinite(row[2]));
}

export function conditionNote(a, b) {
  const label = v => ({ fixed:'고정 촬영', guided:'따라가며 촬영' })[v.capture_mode] ?? '촬영 방식 미기록';
  const extractor = v => {
    const info = v.analysis.provenance?.extractor;
    return [info?.name, info?.delegate].filter(Boolean).join(' ') || '추출기 미기록';
  };
  const notes = [];
  if (!a.capture_mode || !b.capture_mode || a.capture_mode !== b.capture_mode)
    notes.push('촬영 방식 · 이전 '+label(b)+' / 이번 '+label(a));
  if (extractor(a) !== extractor(b))
    notes.push('관절 추출 설정 · 이전 '+extractor(b)+' / 이번 '+extractor(a));
  if (!a.tracking_policy_id || !b.tracking_policy_id) {
    const missing = [!b.tracking_policy_id && '이전', !a.tracking_policy_id && '이번'].filter(Boolean);
    notes.push(missing.join('·')+' 기록에 세부 관절 추출 설정이 없습니다.');
  } else if (a.tracking_policy_id !== b.tracking_policy_id && extractor(a) === extractor(b)) {
    notes.push('관절 추출 설정이 다릅니다 · 이전 '+b.tracking_policy_id+' / 이번 '+a.tracking_policy_id);
  }
  return notes;
}

/** 같은 대상자의 엄격히 이전 기록을 최근순으로, 같은 원본 영상(재업로드·재분석)은 한 번만 남긴다. */
export function priorVisits(current, priors) {
  const source = v => [v.source_sha256, v.analysis.source_id].filter(x => x && x !== 'unknown');
  const seenSources = new Set(source(current));
  const ordered = priors.filter(v => v.subject_id === current.subject_id && v.id !== current.id &&
    Date.parse(v.captured_at) < Date.parse(current.captured_at)).sort((a,b) =>
    Date.parse(b.captured_at) - Date.parse(a.captured_at) || String(a.id).localeCompare(String(b.id)));
  return ordered.filter(v => {
    const ids = source(v);
    if (!ids.length || ids.some(id => seenSources.has(id))) return false;
    ids.forEach(id => seenSources.add(id));
    return true;
  });
}

export function directionHistory(current, priors) {
  const unique = priorVisits(current, priors);
  const groups = [];
  let sharedDirection = false;
  for (const direction of directions) {
    const currentSegments = current.analysis.segments.filter(s => s.direction === direction).length;
    if (!currentSegments) continue;
    const now = directionMetrics(current, direction);
    for (const previous of unique) {
      const previousSegments = previous.analysis.segments.filter(s => s.direction === direction).length;
      if (!previousSegments) continue;
      sharedDirection = true;
      const old = new Map(directionMetrics(previous, direction).map(row => [row[0], row]));
      const rows = now.filter(([key]) => old.has(key)).map(([key,label,value,unit,count]) => ({
        key,label,unit,current:value,previous:old.get(key)[2],currentCount:count,previousCount:old.get(key)[4]
      }));
      if (!rows.length) continue;
      groups.push({direction,currentSegments,previousSegments,previousDate:previous.captured_at,previousId:previous.id,
        notes:conditionNote(current,previous),rows});
      break;
    }
  }
  const sentence = groups.length ? '각 방향에서 비교할 값이 있는 가장 최근 영상과 비교했습니다.' :
    !unique.length ? '이전 기록에서 이번 영상과 다른 원본을 확인할 수 없습니다.' :
    sharedDirection ? '같은 방향의 구간은 있지만 두 기록에서 함께 확인된 걸음 또는 무릎 측정값이 없습니다.' :
    '이번 영상과 이전 영상에서 같은 방향으로 걸은 구간을 찾지 못했습니다.';
  return {sentence,groups};
}
