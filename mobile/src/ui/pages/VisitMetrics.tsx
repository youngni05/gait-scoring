/**
 * 분석 결과지 — 실제 분석 기록(live) 화면.
 * 계산(종합값·걸음 추정·무릎 움직임 폭·지난 기록 해석)은 data/ 의 함수가 맡고, 이 파일은 보여주는 방법만 정한다.
 *
 * 화면 순서
 *   1) 이번 촬영 종합: 구간·방향·좌우를 모은 최종값 하나씩(GDI, 분당 걸음 수, 좌우 걸음 시간, 무릎 움직임 폭)과 결과 해설,
 *      지난 기록이 있으면 변화 한 줄 요약
 *   2) 구간별 상세: 종합값의 근거. 걷는 방향·구간을 고르면 바뀌는 값(GDI, 무릎 각도 그래프, 측정 상세)
 *   지난 기록 비교: 공통 걷기 방향으로 다시 계산한 최종값끼리의 변화·추이와 해석
 */
import { useState, type ReactNode } from 'react';
import { JointChart } from '../components/JointChart';
import { LegendLR } from '../components/layout';
import { LoadingMark } from '../components/brand';
import { ChevronRightIcon, ExpandIcon, InfoIcon, RotateIcon, TriangleAlertIcon } from '../components/icons';
import { dataSource, fmtDate, type ResultRecord } from '../data';
import { postJson } from '../data/analysis';
import { cardMetric, directionLabel, prepareKneeSeries } from '../data/report-values.mjs';
import { directionHistory } from '../data/visit-walking.mjs';
import { EXAMPLE_BAND, describeVisit, fmtDay, visitHistory, visitSummary, type Trend, type Verdict, type VisitSummary }
  from '../data/visit-summary.mjs';
import { useAsync } from '../session';
import '../report.css';

type Visit = NonNullable<ResultRecord['live']>;
type Side = 'left' | 'right';
const SIDES: Side[] = ['left', 'right'];
const sideName = (side: string) => (side === 'left' ? '왼쪽' : '오른쪽');
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const fmt = (n: unknown, digits = 1) => (finite(n) ? n.toFixed(digits) : '측정 보류');
const sec = (ms: number) => (ms / 1000).toFixed(2);
const span = (start: number, stop: number) => `${sec(start)}~${sec(stop)}초`;
export const signed = (v: number, digits: number) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(digits);
export const VERDICT: Record<Verdict, string> = { better: '좋아짐', worse: '나빠짐', similar: '비슷함' };
export const SHORT: Record<string, string> = { gdi: 'GDI', cadence: '걸음 수', stepGap: '좌우 걸음 시간 차', kneeLeft: '좌 무릎 폭', kneeRight: '우 무릎 폭' };
export const BAND_NOTE = `좌우 차이와 변화 판정은 측정 오차를 고려한 예시 기준(GDI ${EXAMPLE_BAND.gdi}점 · 분당 걸음 ${EXAMPLE_BAND.cadence}걸음 · 무릎 ${EXAMPLE_BAND.knee}° · 걸음 시간 ${EXAMPLE_BAND.stepGap}초)입니다. 이보다 작은 차이는 측정 오차와 구분할 수 없어 ‘비슷함’으로 표시합니다. 같은 사람을 다시 촬영한 자료로 확정해야 하는 값입니다.`;

export function kneeCurve(visit: Visit, index: number) {
  const s = visit.analysis.segments[index];
  if (!s) return null;
  const c = prepareKneeSeries(s, visit.curve_time_ms_by_segment);
  return c.available ? c : null;
}

// ── 작은 부품 ─────────────────────────────────

function SectionHead({ title, caption }: { title: string; caption?: string }) {
  return (
    <div className="rs-head">
      <h3>{title}</h3>
      {caption && <p>{caption}</p>}
    </div>
  );
}

function Disclosure({ title, hint, children, inline }: { title: string; hint?: string; children: ReactNode; inline?: boolean }) {
  return (
    <details className={`disc${inline ? ' inline' : ''}`}>
      <summary>
        {title}
        {hint && <small>{hint}</small>}
      </summary>
      <div className="disc-body">{children}</div>
    </details>
  );
}

/** 화살표는 값이 커졌는지·작아졌는지, 색은 좋아짐(초록)·나빠짐(빨강)·비슷함(회색)을 뜻한다. */
export function VerdictIcon({ verdict, delta, size = 28 }: { verdict: Verdict; delta: number; size?: number }) {
  const path = verdict === 'similar' ? 'M7 10h10M7 14h10' : delta > 0 ? 'M12 18V6M6.5 11.5L12 6l5.5 5.5' : 'M12 6v12M6.5 12.5L12 18l5.5-5.5';
  return (
    <span className={`vi ${verdict}`} style={{ width: size, height: size }} role="img" aria-label={VERDICT[verdict]}>
      <svg width={size * 0.62} height={size * 0.62} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.8} strokeLinecap="round" strokeLinejoin="round">
        <path d={path} />
      </svg>
    </span>
  );
}

/**
 * 좋아짐·나빠짐·비슷함 칸에 지표를 나눠 담는다. 글을 읽지 않아도 어디가 바뀌었는지 보이게 한다.
 * 지표가 없는 칸은 그리지 않는다. 모두 비슷하면 '모두 비슷함' 한 칸만 남는다.
 */
function VerdictBoard({ rows }: { rows: { key: string; short: string; delta: number; digits: number; unit: string; verdict: Verdict }[] }) {
  const columns = (['better', 'worse', 'similar'] as const)
    .map((v) => ({ v, items: rows.filter((r) => r.verdict === v) }))
    .filter((c) => c.items.length);
  const allSimilar = columns.length === 1 && columns[0]!.v === 'similar';
  return (
    <div className="board">
      {columns.map(({ v, items }) => (
        <div key={v} className={`board-col ${v}${allSimilar ? ' all' : ''}`}>
          <p className="bh">{allSimilar ? '모두 비슷함' : VERDICT[v]}<b>{items.length}</b></p>
          <ul>
            {items.map((r) => (
              <li key={r.key}>
                <VerdictIcon verdict={r.verdict} delta={r.delta} size={20} />
                <span>{r.short}<small>{signed(r.delta, r.digits)}{r.unit}</small></span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

/**
 * 처음 비교 기록부터 이번 기록까지 좋아지거나 나빠진 지표. 바로 지난 기록과는 '비슷함'이어도
 * 방문을 거듭하며 쌓인 변화가 여기에 드러난다. 판정 폭은 지난 기록 비교와 같은 예시 기준이다.
 * 비교 가능한 지난 기록이 두 번 이상일 때만 의미가 있어, 그 전에는 그리지 않는다.
 */
function SinceFirst({ trends }: { trends: Trend[] }) {
  const moved = trends.filter((t) => t.since && t.since.verdict !== 'similar');
  if (!moved.length) return null;
  const first = moved.map((t) => t.since!.date).sort()[0];
  return (
    <div className="since">
      <p className="since-h">처음 기록({fmtDay(first)})부터 쌓인 변화</p>
      <ul>
        {moved.map((t) => (
          <li key={t.key} className={t.since!.verdict}>
            <VerdictIcon verdict={t.since!.verdict} delta={t.since!.delta} size={20} />
            <span>
              {SHORT[t.key] ?? t.label} {VERDICT[t.since!.verdict]}
              <small>{signed(t.since!.delta, t.digits)}{t.unit}{t.steady && ` · ${t.points.length}회 연속 ${t.delta > 0 ? '↑' : '↓'}`}</small>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** 방문별 값의 흐름. 마지막 점이 이번 기록이다. */
function Spark({ points }: { points: { value: number }[] }) {
  if (points.length < 2) return null;
  const w = 72, h = 28, pad = 4;
  const values = points.map((p) => p.value);
  const lo = Math.min(...values), hi = Math.max(...values);
  const x = (i: number) => pad + (i * (w - 2 * pad)) / (points.length - 1);
  const y = (v: number) => (hi === lo ? h / 2 : h - pad - ((v - lo) * (h - 2 * pad)) / (hi - lo));
  const last = points.length - 1;
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
      {values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={i === last ? 3.5 : 2.2} className={i === last ? 'now' : ''} />)}
    </svg>
  );
}

// ── 지난 기록 불러오기(종합 요약과 비교 탭이 함께 쓴다) ──

function usePriorVisits(record: ResultRecord) {
  return useAsync<{ prior: Visit[]; sentence?: string }>(async (signal) => {
    const current = record.live!;
    const time = (v: Visit) => Date.parse(v.captured_at ?? '');
    const currentTime = time(current);
    if (!Number.isFinite(currentTime)) return { prior: [], sentence: '이번 기록의 촬영일이 없어 이전 방문과 순서를 비교할 수 없습니다.' };
    const records = await dataSource.listResults(record.subjectId, signal);
    const others = records.flatMap((r) => (r.id !== record.id && r.live ? [r.live] : []));
    if (!others.length) return { prior: [], sentence: '이 기기에 저장된 이 대상자의 다른 분석 기록이 없습니다.' };
    const dated = others.filter((v) => Number.isFinite(time(v)));
    const prior = dated.filter((v) => time(v) < currentTime);
    if (!prior.length) {
      const reasons = [];
      if (dated.some((v) => time(v) === currentTime)) reasons.push('같은 촬영 시점으로 저장된 기록은 선후를 구분할 수 없습니다.');
      if (dated.some((v) => time(v) > currentTime)) reasons.push('이 기록보다 나중에 촬영한 결과는 이후 방문의 결과지에서 비교할 수 있습니다.');
      if (dated.length < others.length) reasons.push('촬영일이 없는 기록은 방문 순서를 알 수 없습니다.');
      return { prior: [], sentence: reasons.join(' ') };
    }
    return { prior };
  }, [record.id]);
}

// ── 이번 결과 ─────────────────────────────────

function Overview({ summary, history, openHistory }: {
  summary: VisitSummary; history: ReturnType<typeof visitHistory> | null; openHistory: () => void;
}) {
  const [gdiInfo, setGdiInfo] = useState(false);
  const { gdi, walking, gap, knee } = summary;
  const lines = describeVisit(summary);
  const recent = history?.recent;
  return (
    <section className="rs" aria-label="이번 촬영 종합">
      <SectionHead title="이번 촬영 종합" caption="여러 구간과 두 걷기 방향의 값을 하나로 모은 결과입니다." />
      <div className="ov">
        <div className="ov-gdi">
          <div className="top">
            <p className="ttl">GDI <span>· 보행 편차 지수</span></p>
            <button className="ibtn" aria-label="GDI 설명" aria-expanded={gdiInfo} onClick={() => setGdiInfo(!gdiInfo)}>
              <InfoIcon size={18} />
            </button>
          </div>
          {finite(gdi.value) ? (
            <>
              <p className="big">{gdi.value.toFixed(1)}<small>점</small></p>
              <p className="sides">왼쪽 {fmt(gdi.left)} · 오른쪽 {fmt(gdi.right)}</p>
            </>
          ) : (
            <p className="big none">측정 보류</p>
          )}
          <p className="cap">
            {gdi.directions.length > 1 ? '두 걷기 방향 종합' : gdi.directions.length ? directionLabel(gdi.directions[0]!) + ' 구간만 사용' : '계산 가능한 구간 없음'}
            {' · 정상100 모델 연구용 추정값'}
          </p>
          {gdiInfo && (
            <div className="info">
              100점은 정상 보행의 평균이고, 10점은 정상집단의 1표준편차입니다. 점수가 낮을수록 정상 보행 패턴과의 차이가 큽니다.
              <br />
              종합 방법: 걷기 방향마다 구간 길이 비중으로 평균 → 두 방향을 같은 비중으로 평균 → 왼쪽·오른쪽 평균.
            </div>
          )}
        </div>

        <dl className="ov-rows">
          <div>
            <dt>분당 걸음 수<small>{walking.count}걸음 기준</small></dt>
            <dd>{walking.cadence === null ? <span className="none">관측 부족</span> : <>약 {walking.cadence.toFixed(0)}<small>걸음/분</small></>}</dd>
          </div>
          <div>
            <dt>좌우 걸음 시간<small>{finite(gap.value) ? `차이 ${Math.abs(gap.value).toFixed(2)}초` : '한 걸음 평균'}</small></dt>
            <dd className="lr">
              {SIDES.map((side) => (
                <span key={side} className={side}>
                  {side === 'left' ? '좌' : '우'} {walking[side].value === null ? <em>—</em> : <>{walking[side].value!.toFixed(2)}<small>초</small></>}
                </span>
              ))}
            </dd>
          </div>
          <div>
            <dt>무릎 움직임 폭<small>굽혔다 편 각도 범위</small></dt>
            <dd className="lr">
              {SIDES.map((side) => (
                <span key={side} className={side}>
                  {side === 'left' ? '좌' : '우'} {knee[side] ? <>{knee[side]!.value.toFixed(1)}<small>°</small></> : <em>—</em>}
                </span>
              ))}
            </dd>
          </div>
        </dl>

        <div className="ov-note">
          <p className="h">결과 해설</p>
          <ul className="brief">
            {lines.map((l) => (
              <li key={l.label + l.text} className={l.notice ? 'notice' : ''}>
                <span className="k">{l.label}</span>
                <span className="t">{l.text}</span>
              </li>
            ))}
          </ul>
          {recent && history && (
            <>
              <SinceFirst trends={history.trends} />
              <p className="h" style={{ marginTop: 12 }}>
                지난 기록({fmtDay(recent.previousDate)})과 비교
                {recent.skipped.length > 0 && <small> · {recent.directions.map(directionLabel).join(', ')} 구간끼리</small>}
              </p>
              <VerdictBoard rows={recent.rows.map((r) => ({ ...r, short: SHORT[r.key] ?? r.label }))} />
              <button className="link-btn" onClick={openHistory}>변화와 추이 자세히 보기 <ChevronRightIcon size={16} /></button>
            </>
          )}
        </div>
        <p className="ov-foot">{BAND_NOTE}</p>
      </div>

      <Disclosure title="GDI를 이렇게 종합했습니다">
        {gdi.byDirection.length ? (
          <>
            {gdi.byDirection.map((d) => (
              <div key={d.direction} className="agg">
                <div className="agg-hd">
                  <span className="dir">{directionLabel(d.direction)}</span>
                  <b>왼쪽 {fmt(d.left)} · 오른쪽 {fmt(d.right)}</b>
                </div>
                <ul className="agg-list">
                  {d.segments.map((s) => (
                    <li key={s.segmentId}>
                      <span className="t">{span(s.start, s.stop)}</span>
                      <span className="bar"><i style={{ width: `${Math.round(s.share * 100)}%` }} /></span>
                      <span className="w">비중 {Math.round(s.share * 100)}%</span>
                      <span>좌 {fmt(s.left)} · 우 {fmt(s.right)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <p className="agg-sum">
              {gdi.byDirection.length > 1 ? '두 방향을 같은 비중으로 평균' : '한 방향 값 그대로'} → 왼쪽 {fmt(gdi.left)} · 오른쪽 {fmt(gdi.right)} → 좌우 평균 <b>{fmt(gdi.value)}점</b>
            </p>
            <p className="muted" style={{ fontSize: 13, lineHeight: 1.55 }}>
              구간 비중은 구간 길이(초)로 정합니다. 긴 보행 한 번이 여러 구간으로 나뉘어도 짧은 조각이 같은 무게를 갖지 않게 하기 위해서입니다. 서로 겹치는 구간은 먼저 나온 구간만 씁니다.
            </p>
          </>
        ) : (
          <p className="muted" style={{ fontSize: 14 }}>GDI를 계산할 수 있는 구간이 없습니다.</p>
        )}
      </Disclosure>

      <Disclosure title="걸음·무릎 측정 근거">
        <p className="muted" style={{ fontSize: 13, lineHeight: 1.55 }}>
          분당 걸음 수는 발이 번갈아 닿은 간격 {walking.count}개로 추정했습니다.
          {finite(gap.value) && ` 좌우 걸음 시간 차이는 양발 걸음이 함께 보인 ${gap.segments}개 구간 안에서 계산했습니다.`}
        </p>
        {walking.count > 0 && (
          <ul className="steplist">
            {walking.intervals.map((s) => (
              <li key={s.start_ms + '-' + s.stop_ms}>
                <span className="dir">{directionLabel(s.direction)}</span>
                <span className="t">{span(s.start_ms, s.stop_ms)}</span>
                <span>{sideName(s.side)} 착지</span>
              </li>
            ))}
          </ul>
        )}
        <dl className="kv">
          {SIDES.map((side) => {
            const row = knee[side];
            return (
              <div key={side}>
                <dt>{sideName(side)} 무릎</dt>
                <dd>
                  {row
                    ? `${row.value.toFixed(1)}° · ${directionLabel(row.direction)} · 영상 ${span(row.start, row.stop)}${row.isolatedGaps ? ' · 빠진 프레임 제외' : !row.full ? ' · 연속으로 보인 일부 구간' : ''}`
                    : '연속으로 보인 구간이 없어 측정 보류'}
                </dd>
              </div>
            );
          })}
        </dl>
      </Disclosure>
    </section>
  );
}

export function LiveMetrics({ record, selected, select, zoom, openHistory }: {
  record: ResultRecord; selected: number; select: (i: number) => void; zoom: () => void; openHistory: () => void;
}) {
  const [gdiInfo, setGdiInfo] = useState(false);
  const priors = usePriorVisits(record);
  const visit = record.live!;
  const segments = visit.analysis.segments;
  const index = segments[selected] ? selected : 0;
  const segment = segments[index];
  if (!segment) {
    return (
      <div className="rs">
        <div className="alert alert-warn compact"><TriangleAlertIcon size={18} /><p>분석 구간이 없습니다.</p></div>
      </div>
    );
  }

  const summary = visitSummary(visit);
  const history = priors.data?.prior.length ? visitHistory(visit, priors.data.prior) : null;

  // 걷는 방향별로 구간 번호를 묶는다.
  const groups = new Map<string, number[]>();
  segments.forEach((s: { direction: string }, i: number) => {
    const group = groups.get(s.direction);
    if (group) group.push(i);
    else groups.set(s.direction, [i]);
  });
  const indices = groups.get(segment.direction)!;
  const dirGdi = summary.gdi.byDirection.find((d) => d.direction === segment.direction);
  const report = visit.analysis.current_report;
  const notes = report?.segments?.find((r: { segment_id: string }) => r.segment_id === segment.segment_id)?.metrics;
  const cadence = cardMetric(segment, report, 'cadence');
  const timing = cardMetric(segment, report, 'difference');
  const knee = kneeCurve(visit, index);
  const local = visit.local_observations_by_segment?.[segment.segment_id] ?? [];
  const held = segment.gait?.local_held_contacts ?? segment.gait?.held_contacts;
  const kneeNote = notes?.find((m: { id: string }) => m.id === 'knee');
  const coverage = kneeNote?.evidence;
  const coverageWarnings = SIDES.flatMap((side) => {
    const seen = coverage?.coverage?.[side]?.valid_frames;
    if (typeof seen !== 'number' || seen >= coverage.source_frame_count) return [];
    return [`${sideName(side)} 다리 관측 부족 · 이 구간 ${coverage.source_frame_count}프레임 중 ${seen}프레임에서 무릎 확인`];
  });
  const gdi = segment.gdi?.models?.normal100;
  const baselineGdi = segment.gdi?.models?.baseline;
  const extractor = visit.analysis.provenance?.extractor;

  return (
    <>
      {/* 1) 이번 촬영 종합 */}
      <Overview summary={summary} history={history} openHistory={openHistory} />

      {/* 2) 구간별 상세 */}
      <section className="rs" aria-label="구간별 상세">
        <SectionHead title="구간별 상세" caption="위 종합값의 근거입니다. 걷는 방향과 구간을 고르면 아래 값이 그 구간 값으로 바뀝니다." />

        <div className="tabs seg-tabs" role="tablist" aria-label="걷는 방향" style={{ gridTemplateColumns: `repeat(${groups.size}, minmax(0, 1fr))` }}>
          {Array.from(groups, ([direction, members]) => (
            <button
              key={direction}
              role="tab"
              aria-selected={direction === segment.direction}
              onClick={() => select(direction === segment.direction ? index : members[0]!)}
            >
              {directionLabel(direction)}
              <small>{members.length}구간</small>
            </button>
          ))}
        </div>

        {indices.length > 1 && (
          <div className="chips" role="radiogroup" aria-label="관측 구간">
            {indices.map((i, position) => (
              <button key={segments[i].segment_id} className="chip" role="radio" aria-checked={i === index} onClick={() => select(i)}>
                {position + 1}구간
                <small>{span(segments[i].start_ms, segments[i].stop_ms)}</small>
              </button>
            ))}
          </div>
        )}

        {coverageWarnings.length > 0 && (
          <div className="alert alert-warn compact" role="status">
            <TriangleAlertIcon size={18} />
            {coverageWarnings.length === 1 ? (
              <p>{coverageWarnings[0]}</p>
            ) : (
              <ul className="notes">{coverageWarnings.map((w) => <li key={w}>{w}</li>)}</ul>
            )}
          </div>
        )}

        {/* GDI */}
        <div className="gdi-card">
          <div className="top">
            <div>
              <p className="ttl">GDI <span>· 선택한 구간</span></p>
              <p className="cap">정상100 모델 · 연구용 추정값</p>
            </div>
            <button className="ibtn" aria-label="GDI 설명" aria-expanded={gdiInfo} onClick={() => setGdiInfo(!gdiInfo)}>
              <InfoIcon size={18} />
            </button>
          </div>
          <dl className="gpair">
            {SIDES.map((side) => (
              <div key={side}>
                <dt>{sideName(side)}</dt>
                {finite(gdi?.value?.[side]) ? (
                  <dd>{fmt(gdi.value[side])}<small>점</small></dd>
                ) : (
                  <dd className="none">측정 보류</dd>
                )}
              </div>
            ))}
          </dl>
          {dirGdi && (
            <div className="sub">
              <div className="row">
                <span>
                  이 방향 종합
                  <small> · {dirGdi.segments.length}구간, 구간 길이 비중</small>
                </span>
                <b>왼쪽 {fmt(dirGdi.left)} · 오른쪽 {fmt(dirGdi.right)}</b>
              </div>
            </div>
          )}
          {gdiInfo && <div className="info">구간마다 가운데 124프레임으로 계산한 값입니다. 결과지 맨 위 GDI는 이 구간 값들을 방향별·좌우로 종합한 값입니다.</div>}
        </div>

        {/* 무릎 각도 그래프 */}
        <article className="chart-card">
          <div className="hd">
            <h4>무릎 각도</h4>
            {knee && (
              <button className="zoom-btn" onClick={zoom}>
                <ExpandIcon size={15} /> 확대
              </button>
            )}
          </div>
          {knee ? (
            <>
              <div className="lg"><LegendLR /></div>
              <JointChart curve={knee} native={knee} height={220} label="무릎" />
            </>
          ) : (
            <p className="empty-chart">이 구간에서 관측 가능한 무릎 각도 곡선이 없습니다.</p>
          )}
        </article>

        {/* 측정 상세 */}
        <Disclosure title="측정 상세">
          <dl className="kv">
            <div><dt>구간</dt><dd>{span(segment.start_ms ?? 0, segment.stop_ms)} · {directionLabel(segment.direction)}</dd></div>
            {Array.isArray(held) && <div><dt>착지 후보 검토</dt><dd>
              {held.length ? <>같은 발이 연속 표시되어 시간 간격을 확인한 뒤 {held.length}개 후보를 계산에서 보류했습니다.
                {' '}{held.map((e: {side: string; source_time_ms: number}) => `${sec(e.source_time_ms)}초 ${sideName(e.side)} 발`).join(' · ')}</>
                : '시간 간격을 기준으로 보류한 착지 후보가 없습니다.'}
            </dd></div>}
            {kneeNote?.sentence && <div><dt>무릎 관측</dt><dd>{kneeNote.sentence}</dd></div>}
            {local.map(
              (r: { side: string; value: number; unit: string; status: string }, i: number) =>
                r.unit === 'deg' &&
                r.status === 'limited_local_observation' && (
                  <div key={i}><dt>일부 관측</dt><dd>{sideName(r.side)} 무릎 · 관측된 일부 구간 움직임 폭 {fmt(r.value)}°</dd></div>
                ),
            )}
            {(cadence.reason ?? cadence.sentence) && <div><dt>걸음 리듬</dt><dd>{cadence.reason ?? cadence.sentence}</dd></div>}
            {(timing.reason ?? timing.sentence) && <div><dt>걸음 시간</dt><dd>{timing.reason ?? timing.sentence}</dd></div>}
            <div>
              <dt>기존 결합 모델 GDI</dt>
              <dd>
                {finite(baselineGdi?.value?.left) || finite(baselineGdi?.value?.right)
                  ? `왼쪽 ${fmt(baselineGdi.value.left)}점 · 오른쪽 ${fmt(baselineGdi.value.right)}점 (참고용)`
                  : '측정 보류'}
              </dd>
            </div>
            <div>
              <dt>관절 추출</dt>
              <dd>
                {extractor?.name === 'OpenPose' ? 'OpenPose BODY_25' : 'MediaPipe Full'} · {extractor?.delegate ?? '실행 장치 미확인'}
              </dd>
            </div>
            <div>
              <dt>프레임</dt>
              <dd>영상 전체 {visit.bridge.sampling.frame_count}프레임 중 자세 미검출 {visit.bridge.sampling.missing_pose_frames}프레임</dd>
            </div>
          </dl>
        </Disclosure>
      </section>
    </>
  );
}

// ── 지난 기록 비교 ─────────────────────────────

function visitInput(v: Visit) {
  return {
    subject_id: v.subject_id, visit_id: v.id, captured_at: v.captured_at,
    records: [{ source_id: v.analysis.source_id, source_sha256: v.source_sha256, analysis_id: v.id, analysis: v.analysis,
      tracking_policy_id: v.tracking_policy_id, sampling_policy_id: v.sampling_policy_id }],
  };
}

function TrendRow({ t, directions }: { t: Trend; directions: string[] }) {
  const narrower = t.since && t.since.directions.length < directions.length;
  return (
    <div className="tr-row">
      <VerdictIcon verdict={t.verdict} delta={t.delta} size={36} />
      <div className="tr-body">
        <div className="tr-top">
          <span className="l">{t.label}</span>
          <span className={`kw ${t.verdict}`}>{t.keyword}</span>
        </div>
        <div className="tr-mid">
          <span className="vals">
            <span className="prev">{t.previous.toFixed(t.digits)}</span>
            <span className="arrow">→</span>
            <b>{t.current.toFixed(t.digits)}</b>
            <small>{t.unit === '걸음' ? '걸음/분' : t.unit}</small>
            <span className="dl">{signed(t.delta, t.digits)}</span>
          </span>
          <Spark points={t.points} />
        </div>
        {(t.since || t.steady) && (
          <div className="tags">
            {t.since && (
              <span className={`tag ${t.since.verdict}`} title={narrower ? `처음 기록은 ${t.since.directions.map(directionLabel).join(', ')} 구간끼리 비교` : undefined}>
                처음({fmtDay(t.since.date).slice(5)}) 대비 {signed(t.since.delta, t.digits)}{t.unit} · {VERDICT[t.since.verdict]}{narrower && ' *'}
              </span>
            )}
            {t.steady && <span className="tag streak">{t.points.length}회 연속 {t.delta > 0 ? '↑' : '↓'}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

export function VisitHistory({ record }: { record: ResultRecord }) {
  const [showDetail, setShowDetail] = useState(false);
  const { data, loading, error, reload } = usePriorVisits(record);

  if (loading) {
    return (
      <section className="rs">
        <p role="status" className="primary" style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 15, fontWeight: 600 }}>
          <LoadingMark size={22} /> 지난 기록과 비교하고 있습니다…
        </p>
      </section>
    );
  }
  if (error) {
    return (
      <section className="rs">
        <div className="alert alert-danger" role="alert">
          <TriangleAlertIcon size={20} />
          <div>
            <p className="title">이전 기록을 비교하지 못했습니다</p>
            <p style={{ marginTop: 4 }}>{error instanceof Error && !(error instanceof TypeError) ? error.message : '팀 서버 연결을 확인해 주세요.'}</p>
          </div>
        </div>
        <button className="btn btn-outline btn-sm" onClick={reload}><RotateIcon size={18} /> 다시 시도</button>
      </section>
    );
  }

  const current = record.live!;
  const prior = data?.prior ?? [];
  const history = prior.length ? visitHistory(current, prior) : null;
  const recent = history?.recent;
  const groups = prior.length ? directionHistory(current, prior).groups : [];
  const detailedPrior = prior.filter(
    (v) => current.capture_mode && v.capture_mode === current.capture_mode && current.tracking_policy_id && v.tracking_policy_id === current.tracking_policy_id,
  );

  return (
    <section className="rs" aria-label="지난 기록 비교">
      <SectionHead title="지난 기록과 비교" caption="두 방문에 공통으로 있는 걷기 방향만으로 값을 다시 계산해, 같은 조건끼리 비교합니다." />
      {!recent && (
        <div className="alert alert-info compact"><InfoIcon size={18} /><p>{data?.sentence ?? history?.sentence}</p></div>
      )}
      {recent && history && (
        <>
          <SinceFirst trends={history.trends} />
          <p className="board-h">지난 기록({fmtDay(recent.previousDate)})과 비교</p>
          <VerdictBoard rows={recent.rows.map((r) => ({ ...r, short: SHORT[r.key] ?? r.label }))} />

          <article className="cmp">
            <div className="cmp-hd">
              <span className="meta">
                지난 기록 {fmtDay(recent.previousDate)} → 이번 {fmtDay(current.captured_at)}
                {history.comparable! > 1 && ` · 비교 가능한 지난 기록 ${history.comparable}회`}
              </span>
              <span style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {recent.directions.map((d) => <span key={d} className="dir">{directionLabel(d)}</span>)}
              </span>
            </div>
            <div className="cmp-rows">{history.trends.map((t) => <TrendRow key={t.key} t={t} directions={recent.directions} />)}</div>
            {recent.skipped.length > 0 && (
              <p className="muted" style={{ fontSize: 13, lineHeight: 1.55 }}>
                이번 기록의 {recent.skipped.map(directionLabel).join(', ')} 구간은 지난 기록에 없어 비교에서 뺐습니다. 그래서 위 이번 값은 결과지 맨 위 종합값과 다를 수 있습니다.
              </p>
            )}
            {recent.notes.length > 0 && (
              <div className="alert alert-info compact">
                <InfoIcon size={18} />
                <div>
                  <p style={{ fontWeight: 700 }}>비교 조건 차이</p>
                  <ul className="notes" style={{ marginTop: 4 }}>{recent.notes.map((note) => <li key={note}>{note}</li>)}</ul>
                </div>
              </div>
            )}
          </article>
          <p className="ov-foot">
            {BAND_NOTE} 작은 그래프는 처음 비교 기록부터 이번 기록까지의 흐름이며, 마지막 점이 이번 기록입니다.
            {history.trends.some((t) => t.since && t.since.directions.length < recent.directions.length) &&
              ' * 표시는 처음 기록에 한쪽 걷기 방향만 있어 그 방향끼리 비교했다는 뜻입니다.'}
          </p>
        </>
      )}

      {groups.length > 0 && (
        <Disclosure title="걷기 방향별 상세 비교">
          {groups.map((group) => (
            <article className="cmp" key={group.direction} style={{ background: 'var(--bg)' }}>
              <div className="cmp-hd">
                <span className="dir" style={{ fontSize: 13, padding: '4px 10px' }}>{directionLabel(group.direction)}</span>
                <span className="meta">이전 {fmtDate(group.previousDate)} · {group.previousSegments}구간 → 이번 {group.currentSegments}구간</span>
              </div>
              <div className="cmp-rows">
                {group.rows.map((row) => {
                  const digits = row.unit === '초' ? 2 : row.unit === '°' ? 1 : 0;
                  const countUnit = row.unit === '°' ? '프레임' : '걸음';
                  return (
                    <div className="cmp-row" key={row.key}>
                      <span className="l">{row.label}</span>
                      <span className="vals">
                        <span className="prev">{fmt(row.previous, digits)}</span>
                        <span className="arrow">→</span>
                        <b>{fmt(row.current, digits)}</b>
                        <small>{row.unit}</small>
                      </span>
                      <span className="n">관측 {row.previousCount} → {row.currentCount}{countUnit}</span>
                    </div>
                  );
                })}
              </div>
            </article>
          ))}
        </Disclosure>
      )}
      {recent && !!detailedPrior.length && (
        <details className="disc" onToggle={(e) => setShowDetail(e.currentTarget.open)}>
          <summary>같은 촬영·분석 조건의 추가 해석</summary>
          <div className="disc-body">{showDetail && <StrictHistory current={current} prior={detailedPrior} />}</div>
        </details>
      )}
    </section>
  );
}

function StrictHistory({ current, prior }: { current: Visit; prior: Visit[] }) {
  const { data, loading, error } = useAsync(async (signal) => {
    const res = await postJson('/history/explain', {
      schema: 'ai-history-request/1', request_id: crypto.randomUUID(), comparison_policy_version: 'personal-history/1',
      data_kind: 'recorded_visits', current_visit: visitInput(current), prior_visits: prior.map(visitInput),
    }, signal);
    if (res.comparison?.current_visit_id !== current.id || res.explanation?.schema !== 'gait-history-explanation/1')
      throw Error('기록을 확인하지 못했습니다.');
    return res.explanation;
  }, [current.id, prior]);
  if (loading) return <p role="status" className="muted" style={{ fontSize: 14 }}>추가 해석 중…</p>;
  if (error) return <p className="muted" style={{ fontSize: 14 }}>추가 해석을 불러오지 못했습니다.</p>;
  return (
    <>
      {data?.sentence && <p className="summary-line">{data.sentence}</p>}
      <dl className="kv">
        {data?.evidence?.map(
          (e: { direction: string; label: string; side: string; display_previous: string; display_current: string; meaning: string }, i: number) => (
            <div key={i}>
              <dt>{e.label}</dt>
              <dd>
                <b>{e.display_previous} → {e.display_current}</b> · {directionLabel(e.direction)}
                <br />
                {e.meaning}
              </dd>
            </div>
          ),
        )}
      </dl>
    </>
  );
}
