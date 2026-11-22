/**
 * 분석 결과지의 '하나로 보는 값'과 지난 기록 해석.
 *
 * GDI 종합 순서(2026-11-20 사용자 결정)
 *   구간 → 걷는 방향: 겹치지 않는 구간을 구간 길이(ms) 비중으로 평균한다.
 *                     긴 보행 한 번이 여러 구간으로 나뉘는 일이 많아, 짧은 조각이 같은 비중을 갖지 않게 한다.
 *   방향 → 다리별:    두 방향을 같은 비중으로 평균한다. 카메라 가까운 다리가 방향마다 바뀌기 때문이다.
 *   다리별 → 최종:    왼쪽·오른쪽 평균.
 * 지난 기록 비교는 두 방문에 공통으로 있는 걷기 방향만으로 양쪽 최종값을 다시 만들어 비교한다.
 * '비슷함' 폭(EXAMPLE_BAND)은 재촬영 자료로 정해야 하는 예시 기준이다.
 */
import { visitKneeSummary } from './report-values.mjs';
import { conditionNote, priorVisits, walkingEstimate } from './visit-walking.mjs';

export const DIRECTIONS = ['image_left', 'image_right'];
const SIDES = ['left', 'right'];
const finite = v => typeof v === 'number' && Number.isFinite(v);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/**
 * 측정 오차를 고려해 '비슷함'으로 볼 폭. 실측 근거가 없는 예시 기준이다.
 * 걸음 시간은 30fps 영상의 한 프레임(약 0.033초) 단위로 재므로, 3프레임(0.1초) 이하 차이는 치우침으로 보지 않는다.
 */
export const EXAMPLE_BAND = { gdi: 5, cadence: 5, knee: 5, stepGap: 0.1 };

const restrict = (visit, directions) => ({ ...visit, analysis: { ...visit.analysis,
  segments: visit.analysis.segments.filter(s => directions.includes(s.direction)) } });

export function visitDirections(visit) {
  return DIRECTIONS.filter(d => visit.analysis.segments.some(s => s.direction === d));
}

// ── 이번 기록 종합 ─────────────────────────────

/** 한 방향의 GDI. 겹치는 구간은 먼저 나온 것만 남긴다(분석 서버와 같은 규칙). */
export function directionGdi(visit, direction) {
  const kept = [], intervals = [];
  for (const s of visit.analysis.segments) {
    if (s.direction !== direction) continue;
    const low = s.start_ms, high = s.stop_ms;
    if (!finite(low) || !finite(high) || high <= low) continue;
    if (intervals.some(([a, b]) => low < b && a < high)) continue;
    intervals.push([low, high]);
    const model = s.gdi?.models?.normal100;
    if (model?.status !== 'available' || !SIDES.every(side => finite(model.value?.[side]))) continue;
    kept.push({ segmentId: s.segment_id, start: low, stop: high, weight: high - low, left: model.value.left, right: model.value.right });
  }
  if (!kept.length) return null;
  const total = kept.reduce((sum, s) => sum + s.weight, 0);
  const avg = side => kept.reduce((sum, s) => sum + s[side] * s.weight, 0) / total;
  return { direction, left: avg('left'), right: avg('right'), total_ms: total,
    segments: kept.map(s => ({ ...s, share: s.weight / total })) };
}

export function visitGdi(visit, directions = visitDirections(visit)) {
  const rows = directions.map(d => directionGdi(visit, d)).filter(Boolean);
  if (!rows.length) return { value: null, left: null, right: null, directions: [], byDirection: [] };
  const left = mean(rows.map(r => r.left)), right = mean(rows.map(r => r.right));
  return { value: (left + right) / 2, left, right, directions: rows.map(r => r.direction), byDirection: rows };
}

/**
 * 왼발−오른발 걸음 시간 차이(초). 양발 걸음이 함께 관측된 같은 구간 안에서만 차이를 만들고,
 * 구간별 차이를 사용한 걸음 수 비중으로 평균한다. 다른 구간의 왼발·오른발 평균끼리 빼지 않는다.
 */
export function stepGap(walking) {
  const bySegment = new Map();
  for (const step of walking.intervals) {
    const list = bySegment.get(step.segmentId) ?? [];
    list.push(step);
    bySegment.set(step.segmentId, list);
  }
  const rows = [];
  for (const steps of bySegment.values()) {
    const left = steps.filter(s => s.side === 'left').map(s => s.seconds);
    const right = steps.filter(s => s.side === 'right').map(s => s.seconds);
    if (left.length && right.length) rows.push({ diff: mean(left) - mean(right), n: left.length + right.length });
  }
  if (!rows.length) return { value: null, count: 0, segments: 0 };
  const n = rows.reduce((sum, r) => sum + r.n, 0);
  return { value: rows.reduce((sum, r) => sum + r.diff * r.n, 0) / n, count: n, segments: rows.length };
}

export function visitSummary(visit, directions = visitDirections(visit)) {
  const scoped = restrict(visit, directions);
  const walking = walkingEstimate(scoped);
  const kneeRows = visitKneeSummary(scoped);
  return { directions, available: visitDirections(visit), gdi: visitGdi(visit, directions), walking,
    gap: stepGap(walking), knee: Object.fromEntries(kneeRows.map(r => [r.side, r])), kneeRows };
}

// ── 문장 도우미 ───────────────────────────────

const signed = (v, digits) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(digits);
/** 받침 유무로 조사를 고른다. 영문 약어(GDI)는 받침 없음으로 읽는다. */
function particle(word, withFinal, withoutFinal) {
  const ch = word.trim().at(-1) ?? '';
  const code = ch.charCodeAt(0) - 0xac00;
  const final = code >= 0 && code <= 11171 ? code % 28 !== 0 : /[LMNRlmnr0136780]$/.test(ch);
  return final ? withFinal : withoutFinal;
}
/** 여러 이름을 쉼표로 잇고 마지막 이름에 맞춰 조사를 붙인다. */
const subjectOf = (items, withFinal, withoutFinal) =>
  items.map(i => i.text).join(', ') + particle(items.at(-1).label, withFinal, withoutFinal);
export const fmtDay = iso => (iso ?? '').slice(0, 10).replaceAll('-', '.');

/**
 * 기록 하나만으로 할 수 있는 해설. 외부 정상범위는 쓰지 않는다(GDI 정의와 좌우 차이만).
 * 숫자는 바로 위 지표에 보이므로 '항목 · 짧은 결론'만 돌려준다. notice=true는 눈여겨볼 결과다.
 */
export function describeVisit(summary) {
  const items = [];
  const add = (label, text, notice = false) => items.push({ label, text, notice });
  const g = summary.gdi;
  if (finite(g.value)) {
    // 화면에 보이는 반올림 값으로 차이를 계산한다. 정상집단 1표준편차(10점) 안이면 해설하지 않는다.
    const shown = Math.round(g.value * 10) / 10, gap = Math.round((100 - shown) * 10) / 10;
    if (gap > 10) add('GDI', `정상 평균보다 ${gap.toFixed(1)}점 낮음 (약 ${(gap / 10).toFixed(1)}표준편차)`, true);
    const sideGap = Math.round(g.left * 10) / 10 - Math.round(g.right * 10) / 10;
    if (Math.abs(sideGap) > EXAMPLE_BAND.gdi) add('GDI 좌우', `${sideGap < 0 ? '왼쪽' : '오른쪽'} 다리가 ${Math.abs(sideGap).toFixed(1)}점 낮음`, true);
    if (summary.available.length > g.directions.length) add('GDI', '한쪽 걷기 방향 구간만 사용');
  } else {
    add('GDI', '계산할 만큼 긴 보행 구간 없음');
  }
  const t = summary.gap;
  if (finite(t.value)) {
    if (Math.abs(t.value) <= EXAMPLE_BAND.stepGap) add('걸음 시간', '좌우 차이 없음 (측정 오차 범위)');
    else add('걸음 시간', `${t.value > 0 ? '왼발' : '오른발'}이 ${Math.abs(t.value).toFixed(2)}초 더 김`, true);
  }
  const { left, right } = summary.knee;
  if (left && right && (!left.full || !right.full || left.segmentId !== right.segmentId)) {
    add('무릎 움직임 폭', '같은 구간의 양쪽 무릎 관측 부족 · 좌우 비교 보류');
  } else if (left && right) {
    const d = left.value - right.value;
    if (Math.abs(d) <= EXAMPLE_BAND.knee) add('무릎 움직임 폭', '좌우 비슷함');
    else add('무릎 움직임 폭', `${d < 0 ? '왼쪽' : '오른쪽'}이 ${Math.abs(d).toFixed(1)}° 작음`, true);
  } else if (left || right) {
    add('무릎 움직임 폭', `${left ? '왼쪽' : '오른쪽'}만 측정됨`);
  }
  if (!items.length) add('좌우', '눈에 띄는 차이 없음');
  return items;
}

// ── 지난 기록과 비교 ───────────────────────────

export const METRICS = [
  { key: 'gdi', label: 'GDI', unit: '점', digits: 1, band: EXAMPLE_BAND.gdi, better: 'up', read: s => s.gdi.value,
    up: '정상 보행 패턴에 더 가까워졌습니다.', down: '정상 보행 패턴과의 차이가 커졌습니다.', upKey: '정상 패턴에 가까워짐', downKey: '정상 패턴과 멀어짐' },
  { key: 'cadence', label: '분당 걸음 수', unit: '걸음', digits: 0, band: EXAMPLE_BAND.cadence, better: 'up', read: s => s.walking.cadence,
    up: '걸음 리듬이 빨라졌습니다.', down: '걸음 리듬이 느려졌습니다.', upKey: '리듬 빨라짐', downKey: '리듬 느려짐' },
  { key: 'stepGap', label: '좌우 걸음 시간 차이', unit: '초', digits: 2, band: EXAMPLE_BAND.stepGap, better: 'down',
    read: s => (finite(s.gap.value) ? Math.abs(s.gap.value) : null),
    up: '걸음 시간이 한쪽으로 더 치우쳤습니다.', down: '좌우 걸음 시간이 더 고르게 되었습니다.', upKey: '한쪽으로 치우침', downKey: '좌우 고르게' },
  { key: 'kneeLeft', label: '왼쪽 무릎 움직임 폭', unit: '°', digits: 1, band: EXAMPLE_BAND.knee, better: 'up',
    read: s => s.knee.left?.value ?? null, up: '무릎을 더 크게 굽혔다 폈습니다.', down: '무릎 움직임 폭이 줄었습니다.', upKey: '더 크게 움직임', downKey: '움직임 줄어듦' },
  { key: 'kneeRight', label: '오른쪽 무릎 움직임 폭', unit: '°', digits: 1, band: EXAMPLE_BAND.knee, better: 'up',
    read: s => s.knee.right?.value ?? null, up: '무릎을 더 크게 굽혔다 폈습니다.', down: '무릎 움직임 폭이 줄었습니다.', upKey: '더 크게 움직임', downKey: '움직임 줄어듦' },
];

export function verdict(delta, metric) {
  if (Math.abs(delta) <= metric.band + 1e-9) return 'similar';
  return (delta > 0) === (metric.better === 'up') ? 'better' : 'worse';
}

export function compareVisits(current, prior) {
  const priorDirections = visitDirections(prior);
  const common = visitDirections(current).filter(d => priorDirections.includes(d));
  if (!common.length) return null;
  const now = visitSummary(current, common), before = visitSummary(prior, common);
  const notes = conditionNote(current, prior);
  const rows = METRICS.flatMap(m => {
    const side = m.key === 'kneeLeft' ? 'left' : m.key === 'kneeRight' ? 'right' : null;
    if (side) {
      const a = now.knee[side], b = before.knee[side];
      if (!a?.full || !b?.full || a.direction !== b.direction) {
        notes.push(`${m.label}은 같은 방향에서 구간 전체를 관측한 두 값이 없어 비교를 보류했습니다.`);
        return [];
      }
    }
    const cur = m.read(now), prev = m.read(before);
    if (!finite(cur) || !finite(prev)) return [];
    const delta = cur - prev, v = verdict(delta, m);
    return [{ key: m.key, label: m.label, unit: m.unit, digits: m.digits, band: m.band, current: cur, previous: prev, delta,
      ...(side ? { kneeDirection: now.knee[side].direction } : {}),
      verdict: v, meaning: v === 'similar' ? `예시 기준(±${m.band}${m.unit}) 안의 차이입니다.` : delta > 0 ? m.up : m.down,
      keyword: v === 'similar' ? '큰 변화 없음' : delta > 0 ? m.upKey : m.downKey }];
  });
  if (!rows.length) return null;
  return { previousId: prior.id, previousDate: prior.captured_at, directions: common,
    skipped: visitDirections(current).filter(d => !common.includes(d)), notes, rows };
}

const withDelta = r => ({ label: r.label, text: `${r.label}(${signed(r.delta, r.digits)}${r.unit})` });
const nameOnly = r => ({ label: r.label, text: r.label });

/** 가장 최근 비교 결과를 한두 문장으로 정리한다. */
function narrate(recent) {
  const by = v => recent.rows.filter(r => r.verdict === v);
  const better = by('better').map(withDelta), worse = by('worse').map(withDelta), similar = by('similar').map(nameOnly);
  const lead = `지난 기록(${fmtDay(recent.previousDate)})과 비교하면 `;
  let first;
  if (better.length && worse.length) first = `${subjectOf(better, '이', '가')} 좋아졌고, ${subjectOf(worse, '은', '는')} 나빠졌습니다.`;
  else if (better.length) first = `${subjectOf(better, '이', '가')} 좋아졌습니다.`;
  else if (worse.length) first = `${subjectOf(worse, '이', '가')} 나빠졌습니다.`;
  else return [lead + '모든 항목에 큰 변화가 없습니다.'];
  const lines = [lead + first];
  if (similar.length) lines.push(`${subjectOf(similar, '은', '는')} 비슷합니다.`);
  return lines;
}

/**
 * 지난 기록 전체 해석. 가장 최근 비교 가능한 방문과의 변화를 먼저 두고,
 * 비교 가능한 방문이 둘 이상이면 처음 기록부터의 흐름(추이)을 지표마다 붙인다.
 */
export function visitHistory(current, priors) {
  const unique = priorVisits(current, priors);
  const comparisons = unique.map(p => compareVisits(current, p)).filter(Boolean);
  if (!comparisons.length) {
    return { recent: null, trends: [], lines: [], visits: unique.length,
      sentence: !unique.length ? '이전 기록에서 이번 영상과 다른 원본을 확인할 수 없습니다.'
        : '지난 기록과 같은 방향으로 걸은 구간 또는 함께 측정된 값이 없어 비교할 수 없습니다.' };
  }
  const recent = comparisons[0];
  const trends = recent.rows.map(row => {
    const sameSeries = r => r.key === row.key && r.kneeDirection === row.kneeDirection;
    const older = [...comparisons].reverse().find(c => c !== recent && c.rows.some(sameSeries));
    const points = [...comparisons].reverse().flatMap(c => {
      const r = c.rows.find(sameSeries);
      return r ? [{ date: c.previousDate, value: r.previous }] : [];
    });
    points.push({ date: current.captured_at, value: row.current, current: true });
    const since = older ? older.rows.find(r => r.key === row.key) : null;
    const steps = points.slice(1).map((p, i) => p.value - points[i].value);
    const steady = points.length >= 3 && since && since.verdict !== 'similar' &&
      (steps.every(d => d > 0) || steps.every(d => d < 0));
    return { ...row, points, steady: Boolean(steady),
      since: since ? { date: older.previousDate, delta: since.delta, verdict: since.verdict, directions: older.directions } : null };
  });
  const lines = narrate(recent);
  const steady = trends.filter(t => t.steady);
  if (steady.length) {
    const ups = steady.filter(t => t.delta > 0), downs = steady.filter(t => t.delta < 0);
    for (const [rows, word] of [[ups, '늘었습니다'], [downs, '줄었습니다']]) {
      if (!rows.length) continue;
      lines.push(`${subjectOf(rows.map(nameOnly), '은', '는')} 최근 ${rows[0].points.length}번의 기록에서 계속 ${word}.`);
    }
  }
  return { recent, trends, lines, visits: unique.length, comparable: comparisons.length,
    sentence: '두 방문에 공통으로 있는 걷기 방향만으로 양쪽 값을 다시 계산해 비교했습니다.' };
}

// ── 대상자 전체 기록 해석 ───────────────────────

/**
 * 한 사람의 기록 전체를 가장 최근 방문 기준으로 해석한다. 계산은 visitHistory와 같다.
 * 지표마다 '처음 비교 가능한 기록 → 가장 최근 기록'의 변화(fromFirst)를 붙인다.
 * 비교 가능한 지난 기록이 하나뿐이면 그 기록이 곧 처음 기록이다.
 * 촬영일이 없는 기록은 방문 순서를 알 수 없어 해석에서 빼고 개수만 알린다.
 */
export function subjectOverview(visits) {
  const time = v => Date.parse(v.captured_at ?? '');
  const dated = visits.filter(v => Number.isFinite(time(v))).sort((a, b) => time(b) - time(a) || String(a.id).localeCompare(String(b.id)));
  const undated = visits.length - dated.length;
  const rows = dated.map(v => {
    const s = visitSummary(v);
    return { id: v.id, date: v.captured_at, gdi: s.gdi.value, cadence: s.walking.cadence, directions: s.directions };
  });
  if (!dated.length) return { latest: null, history: null, trends: [], visits: rows, undated, notes: [], lines: [] };
  const latest = dated[0];
  const priors = dated.slice(1);
  const history = visitHistory(latest, priors);
  const recent = history.recent;
  const trends = history.trends.map(t => ({ ...t,
    fromFirst: t.since ?? { date: recent.previousDate, delta: t.delta, verdict: t.verdict, directions: recent.directions } }));
  // 어느 방문의 조건이 다른지 알 수 있게 지난 기록의 촬영일을 앞에 붙인다.
  const notes = priorVisits(latest, priors).flatMap(p => conditionNote(latest, p).map(n => `${fmtDay(p.captured_at)} 기록 — ${n}`));
  return { latest, history, trends, visits: rows, undated, notes, lines: overviewLines(trends, priors.length) };
}

/** 전체 기록 해석의 머리 문장. 처음 기록 대비 변화 → 연속 변화 순서로 쓴다. */
function overviewLines(trends, priorCount) {
  if (!trends.length) {
    return [priorCount ? '지난 기록과 같은 방향으로 걸은 구간이 없어 변화를 계산하지 못했습니다.'
      : '기록이 2회 이상 쌓이면 방문 사이의 변화를 보여 드립니다.'];
  }
  const first = trends.map(t => t.fromFirst.date).sort()[0];
  const by = v => trends.filter(t => t.fromFirst.verdict === v)
    .map(t => ({ label: t.label, text: `${t.label}(${signed(t.fromFirst.delta, t.digits)}${t.unit})` }));
  const better = by('better'), worse = by('worse');
  const lead = `처음 기록(${fmtDay(first)})과 비교하면 `;
  const lines = [];
  if (better.length && worse.length) lines.push(lead + `${subjectOf(better, '이', '가')} 좋아졌고, ${subjectOf(worse, '은', '는')} 나빠졌습니다.`);
  else if (better.length) lines.push(lead + `${subjectOf(better, '이', '가')} 좋아졌습니다.`);
  else if (worse.length) lines.push(lead + `${subjectOf(worse, '이', '가')} 나빠졌습니다.`);
  else lines.push(lead + '모든 항목이 비슷한 범위 안에 있습니다.');
  for (const t of trends.filter(x => x.steady)) {
    lines.push(`${t.label}${particle(t.label, '은', '는')} 최근 ${t.points.length}번의 기록에서 계속 ${t.delta > 0 ? '늘었습니다' : '줄었습니다'}.`);
  }
  return lines;
}
