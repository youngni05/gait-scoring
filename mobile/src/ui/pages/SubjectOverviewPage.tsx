/**
 * 03-1 대상자 전체 기록 해석.
 * 결과지 하나가 아니라 한 사람의 방문 기록 전체를 가장 최근 방문 기준으로 본다.
 * 계산은 data/visit-summary.mjs의 subjectOverview(= 결과지의 지난 기록 비교와 같은 계산)가 맡고,
 * 이 화면은 ① 한 줄 해석 ② 지표별 흐름 그래프 ③ 방문 목록 ④ 비교 조건 안내를 보여 준다.
 */
import { PageError, PageLoading, TopBar } from '../components/layout';
import { ChevronRightIcon, InfoIcon, VideoIcon } from '../components/icons';
import { dataSource, fmtDate, fmtWeekday } from '../data';
import { EXAMPLE_BAND, fmtDay, subjectOverview, type OverviewTrend, type SubjectOverview } from '../data/visit-summary.mjs';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { BAND_NOTE, SHORT, VERDICT, VerdictIcon, signed } from './VisitMetrics';
import { NotFound } from './NotFound';
import '../report.css';

const shortDay = (iso: string) => fmtDay(iso).slice(5);
const unitText = (unit: string) => (unit === '걸음' ? '걸음/분' : unit);
const BAND: Record<string, number> = {
  gdi: EXAMPLE_BAND.gdi, cadence: EXAMPLE_BAND.cadence, stepGap: EXAMPLE_BAND.stepGap, kneeLeft: EXAMPLE_BAND.knee, kneeRight: EXAMPLE_BAND.knee,
};

export function SubjectOverviewPage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(
    (signal) => Promise.all([dataSource.getSubject(subjectId, signal), dataSource.listResults(subjectId, signal)]),
    [subjectId],
  );
  const back = () => navigate(`/subjects/${subjectId}`);
  if (loading) return <PageLoading title="전체 기록 해석" nav="back" onNav={back} />;
  if (error || !data) return <PageError title="전체 기록 해석" onNav={back} onRetry={reload} />;
  const [subject, records] = data;
  if (!subject) return <NotFound />;

  const overview = subjectOverview(records.flatMap((r) => (r.live ? [r.live] : [])));
  const open = (id: string) => navigate(`/subjects/${subject.id}/results/${id}`);

  return (
    <div className="screen" style={{ paddingBottom: 40 }}>
      <TopBar title="전체 기록 해석" nav="back" onNav={back} subject={subject} />
      {overview.latest ? (
        <>
          <Headline overview={overview} />
          {overview.trends.length > 0 && <Trends trends={overview.trends} />}
          <Visits overview={overview} open={open} />
          <Notes overview={overview} />
        </>
      ) : (
        <section className="empty">
          <h2>해석할 분석 기록이 없습니다</h2>
          <p>
            {overview.undated > 0 ? '촬영일이 없는 기록은 방문 순서를 알 수 없어 해석하지 않습니다.' : `${subject.name} 님의 보행 영상을 분석하면`}
            <br />
            {overview.undated > 0 ? '기록의 촬영일을 확인해 주세요.' : '이곳에서 방문 사이의 변화를 볼 수 있습니다.'}
          </p>
          <button className="btn btn-primary" style={{ marginTop: 40 }} onClick={() => navigate(`/subjects/${subject.id}/analyze`)}>
            <VideoIcon /> 새 보행 영상 분석
          </button>
        </section>
      )}
    </div>
  );
}

/** 기간·횟수와 한 줄 해석, 처음 기록 대비 좋아지거나 나빠진 지표 칩. */
function Headline({ overview }: { overview: SubjectOverview }) {
  const { visits, trends, lines } = overview;
  const first = visits.at(-1)!, last = visits[0]!;
  const days = Math.round((Date.parse(last.date) - Date.parse(first.date)) / 86_400_000);
  const moved = trends.filter((t) => t.fromFirst.verdict !== 'similar');
  return (
    <section className="rs" aria-label="한눈에 보기">
      <div className="ov-note" style={{ borderRadius: 'var(--r-lg)' }}>
        <p className="h">
          기록 {visits.length}회
          {visits.length > 1 && <small> · {fmtDay(first.date)} ~ {fmtDay(last.date)} ({days}일)</small>}
        </p>
        <ul className="ovw-lines">
          {lines.map((l) => <li key={l}>{l}</li>)}
        </ul>
        {moved.length > 0 && (
          <div className="since" style={{ background: 'var(--bg)' }}>
            <ul style={{ marginTop: 0 }}>
              {moved.map((t) => (
                <li key={t.key} className={t.fromFirst.verdict}>
                  <VerdictIcon verdict={t.fromFirst.verdict} delta={t.fromFirst.delta} size={20} />
                  <span>
                    {SHORT[t.key] ?? t.label} {VERDICT[t.fromFirst.verdict]}
                    <small>{signed(t.fromFirst.delta, t.digits)}{t.unit}</small>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

function Trends({ trends }: { trends: OverviewTrend[] }) {
  return (
    <section className="rs" aria-label="지표별 흐름">
      <div className="rs-head">
        <h3>지표별 흐름</h3>
        <p>점 하나가 방문 한 번입니다. 회색 띠는 처음 기록에서 ‘비슷함’으로 보는 범위이며, 띠를 벗어나면 처음보다 좋아지거나 나빠진 것입니다.</p>
      </div>
      {trends.map((t) => (
        <article key={t.key} className="ovw-card">
          <div className="ovw-top">
            <span className="l">{t.label}</span>
            <span className={`kw ${t.fromFirst.verdict}`}>
              처음 대비 {VERDICT[t.fromFirst.verdict]}
            </span>
          </div>
          <p className="ovw-vals">
            <span className="prev">{t.points[0]!.value.toFixed(t.digits)}</span>
            <span className="arrow">→</span>
            {/* 처음 기록과의 비교에서 다시 계산한 이번 값. 화살표 양쪽과 차이가 서로 맞도록 같은 비교의 값을 쓴다. */}
            <b>{(t.points[0]!.value + t.fromFirst.delta).toFixed(t.digits)}</b>
            <small>{unitText(t.unit)}</small>
            <span className="dl">{signed(t.fromFirst.delta, t.digits)}</span>
            {t.steady && <span className="tag streak">{t.points.length}회 연속 {t.delta > 0 ? '↑' : '↓'}</span>}
          </p>
          <TrendChart trend={t} band={BAND[t.key] ?? 0} />
        </article>
      ))}
    </section>
  );
}

/**
 * 방문별 값의 흐름. 가로는 방문 순서(간격은 같게), 세로는 값이다.
 * 처음 값 ± 예시 기준을 회색 띠로 깔아 '처음과 비슷한 범위'를 보여 준다.
 */
function TrendChart({ trend, band }: { trend: OverviewTrend; band: number }) {
  const pts = trend.points;
  const W = 320, H = 132, L = 12, R = 12, T = 22, B = 24;
  const base = pts[0]!.value;
  const values = pts.map((p) => p.value);
  let lo = Math.min(...values, base - band), hi = Math.max(...values, base + band);
  const pad = (hi - lo || 1) * 0.12;
  lo -= pad; hi += pad;
  const x = (i: number) => (pts.length === 1 ? W / 2 : L + (i * (W - L - R)) / (pts.length - 1));
  const y = (v: number) => T + ((hi - v) * (H - T - B)) / (hi - lo);
  const labelAll = pts.length <= 5;
  const last = pts.length - 1;
  return (
    <svg className="ovw-chart" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`${trend.label} ${pts.map((p) => `${shortDay(p.date)} ${p.value.toFixed(trend.digits)}`).join(', ')}`}>
      <rect className="band" x={0} width={W} y={y(base + band)} height={y(base - band) - y(base + band)} rx={6} />
      <polyline points={pts.map((p, i) => `${x(i)},${y(p.value)}`).join(' ')} />
      {pts.map((p, i) => (
        <g key={p.date + i}>
          <circle cx={x(i)} cy={y(p.value)} r={i === last ? 5 : 3.5} className={i === last ? 'now' : ''} />
          {(labelAll || i === 0 || i === last) && (
            <>
              <text className={`val${i === last ? ' now' : ''}`} x={x(i)} y={y(p.value) - 9} textAnchor={i === 0 ? 'start' : i === last ? 'end' : 'middle'}>
                {p.value.toFixed(trend.digits)}
              </text>
              <text className="day" x={x(i)} y={H - 6} textAnchor={i === 0 ? 'start' : i === last ? 'end' : 'middle'}>{shortDay(p.date)}</text>
            </>
          )}
        </g>
      ))}
    </svg>
  );
}

function Visits({ overview, open }: { overview: SubjectOverview; open: (id: string) => void }) {
  return (
    <section className="rs" aria-label="방문 기록">
      <div className="rs-head"><h3>방문 기록</h3></div>
      <ul className="ovw-visits">
        {overview.visits.map((v, i) => (
          <li key={v.id}>
            <button onClick={() => open(v.id)}>
              <span className="d">
                {fmtDate(v.date)} <small>{fmtWeekday(v.date)}{i === 0 && ' · 최신'}</small>
              </span>
              <span className="m">GDI <b>{v.gdi === null ? '—' : v.gdi.toFixed(1)}</b></span>
              <span className="m">걸음 <b>{v.cadence === null ? '—' : v.cadence.toFixed(0)}</b></span>
              <ChevronRightIcon size={16} className="chev" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Notes({ overview }: { overview: SubjectOverview }) {
  return (
    <section className="rs" aria-label="해석 기준">
      {overview.notes.length > 0 && (
        <div className="alert alert-info compact">
          <InfoIcon size={18} />
          <div>
            <p style={{ fontWeight: 700 }}>촬영·분석 조건이 다른 기록이 있습니다</p>
            <ul className="notes" style={{ marginTop: 4 }}>{overview.notes.map((n) => <li key={n}>{n}</li>)}</ul>
          </div>
        </div>
      )}
      {overview.trends.length > 0 && <p className="ov-foot">
        변화는 가장 최근 기록({fmtDay(overview.latest!.captured_at)})을 기준으로, 각 지난 기록과 두 방문에 공통으로 있는 걷기 방향만으로 다시 계산한 값입니다.
        그래서 그래프의 지난 값은 방문 목록의 종합값과 다를 수 있습니다.
        {overview.undated > 0 && ` 촬영일이 없는 기록 ${overview.undated}건은 순서를 알 수 없어 뺐습니다.`} {BAND_NOTE}
      </p>}
      {overview.trends.length === 0 && overview.undated > 0 && (
        <p className="ov-foot">촬영일이 없는 기록 {overview.undated}건은 순서를 알 수 없어 뺐습니다.</p>
      )}
    </section>
  );
}
