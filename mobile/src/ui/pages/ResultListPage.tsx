/** 05 이전 결과 목록 · 05-1 이전 결과 없음 */
import { PageError, PageLoading, TopBar } from '../components/layout';
import { ChevronRightIcon, SortIcon, TriangleAlertIcon, UserIcon, VideoIcon } from '../components/icons';
import { dataSource, fmtDate, fmtTime, fmtWeekday } from '../data';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { NotFound } from './NotFound';

export function ResultListPage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(
    (signal) => Promise.all([dataSource.getSubject(subjectId, signal), dataSource.listResults(subjectId, signal)]),
    [subjectId],
  );
  const back = () => navigate(`/subjects/${subjectId}`);
  if (loading) return <PageLoading title="이전 결과" nav="back" onNav={back} />;
  if (error || !data) return <PageError title="이전 결과" onNav={back} onRetry={reload} />;
  const [subject, records] = data;
  if (!subject) return <NotFound />;

  if (records.length === 0) {
    return (
      <div className="screen">
        <TopBar title="이전 결과" nav="back" onNav={back} subject={subject} sticky={false} />
        <section className="empty">
          <div className="empty-icon" style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--fg-40)' }}>
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
              <path d="M6 14l1.5-3a2 2 0 011.8-1H20a1 1 0 011 1.3l-1.6 6.2A2 2 0 0117.5 19H4a2 2 0 01-2-2V5a2 2 0 012-2h4l2 3h7a2 2 0 012 2v2" />
            </svg>
          </div>
          <h2>아직 측정 기록이 없습니다</h2>
          <p>
            {subject.name} 님의 보행 영상을 분석하면
            <br />
            이곳에 결과가 최신순으로 표시됩니다.
          </p>
          <button className="btn btn-primary" style={{ marginTop: 40 }} onClick={() => navigate(`/subjects/${subject.id}/analyze`)}>
            <VideoIcon /> 새 보행 영상 분석
          </button>
        </section>
      </div>
    );
  }

  return (
    <div className="screen" style={{ paddingBottom: 40 }}>
      <TopBar title="이전 결과" nav="back" onNav={back} subject={subject} />

      <div style={{ padding: '20px 20px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ fontSize: 16, fontWeight: 700 }}>
          측정 기록 <span className="primary">{records.length}건</span>
        </p>
        <p style={{ fontSize: 14, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 4 }} className="muted">
          <SortIcon size={16} /> 최신순
        </p>
      </div>

      <ul style={{ padding: '0 20px', display: 'grid', gap: 12 }}>
        {records.map((r, i) => {
          const isLatest = i === 0;
          const partial = Object.keys(r.missing).length > 0;
          return (
            <li key={r.id}>
              <button
                className="record-btn"
                data-latest={isLatest || undefined}
                onClick={() => navigate(`/subjects/${subject.id}/results/${r.id}`)}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
                  <div>
                    <p style={{ fontSize: 19, fontWeight: 800 }}>
                      {fmtDate(r.measuredAt)}{' '}
                      <span style={{ fontSize: 15, fontWeight: 600 }} className="muted">({fmtWeekday(r.measuredAt)})</span>
                    </p>
                    <p style={{ fontSize: 14, marginTop: 2 }} className="muted">{fmtTime(r.measuredAt)} {r.dateBasis === 'legacy_capture' ? '촬영 (기존 기록)' : '분석 시작'}</p>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    {isLatest && <span className="pill pill-primary">최신</span>}
                    {r.status === 'not_available' && <span className="pill">분석 결과 없음</span>}
                    {partial && (
                      <span className="pill pill-warn"><TriangleAlertIcon size={14} />일부 결과 누락</span>
                    )}
                  </div>
                </div>
                <div className="metric-grid3" style={{ marginTop: 16 }}>
                  <div>
                    <p className="k">GDI</p>
                    <p className={`v ${r.gdi === null ? 'none' : ''}`}>{r.gdi ?? '—'}</p>
                  </div>
                  <div>
                    <p className="k">케이던스</p>
                    <p className={`v ${r.cadence === null ? 'none' : ''}`}>
                      {r.cadence ?? '—'}
                      {r.cadence !== null && <small>걸음/분</small>}
                    </p>
                  </div>
                  <div>
                    <p className="k">스트라이드(cm)</p>
                    <p className={`v ${r.strideLength == null ? 'none' : ''}`}>{r.strideLength ?? '—'}</p>
                  </div>
                </div>
                <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 14, color: 'var(--fg-70)' }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <UserIcon size={15} /> {r.dateBasis === 'legacy_capture' ? '촬영 당시' : '분석 시작일 기준'} {r.ageAtCapture}세 · 현재 키 {r.heightAtCapture ?? '—'}cm
                  </span>
                  {isLatest ? (
                    <span className="primary" style={{ fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                      결과지 보기 <ChevronRightIcon size={14} />
                    </span>
                  ) : (
                    <ChevronRightIcon size={18} className="chev" />
                  )}
                </div>
              </button>
            </li>
          );
        })}
      </ul>
      <p style={{ padding: '24px 20px 0', textAlign: 'center', fontSize: 14, color: 'var(--fg-50)' }}>모든 기록을 불러왔습니다.</p>
    </div>
  );
}
