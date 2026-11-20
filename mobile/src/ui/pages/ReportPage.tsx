/**
 * 07 분석 결과지 · 07-1 결과지 일부 누락.
 * 새 분석 직후와 이전 기록 조회 모두 이 화면 하나를 쓴다(디자인 원칙: 결과지는 같은 모양).
 * 구성: 1) 대상자 정보(촬영 당시 값) → 2) 핵심 지표 → 3) 관절별 그래프 → 다운로드.
 */
import { useState } from 'react';
import { DownloadSheet } from '../components/DownloadSheet';
import { JointChart } from '../components/JointChart';
import { AxisNote, LegendLR, PageError, PageLoading, TopBar } from '../components/layout';
import { ChartLineIcon, DownloadIcon, ExpandIcon, InfoIcon, TriangleAlertIcon } from '../components/icons';
import { JOINTS, JOINT_LABEL, MISSING_LABEL, dataSource, fmtDate, fmtTime, fmtWeekday, type MissingKey } from '../data';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { LiveMetrics, VisitHistory } from './VisitMetrics';
import { NotFound } from './NotFound';

export function ReportPage() {
  const { subjectId = '', recordId = '' } = useParams();
  const navigate = useNavigate();
  const [openInfo, setOpenInfo] = useState<{ gdi?: boolean; sym?: boolean }>({});
  const [history,setHistory]=useState(false);
  const [selected,setSelected]=useState(()=>Number(sessionStorage.getItem('segment-'+recordId)??0));
  const [downloading, setDownloading] = useState(false);
  const { data, loading, error, reload } = useAsync(
    (signal) => Promise.all([dataSource.getSubject(subjectId, signal), dataSource.getResult(subjectId, recordId, signal)]),
    [subjectId, recordId],
  );
  const toList = () => navigate(`/subjects/${subjectId}/results`);
  if (loading) return <PageLoading title="분석 결과지" nav="back" onNav={toList} />;
  if (error || !data) return <PageError title="분석 결과지" onNav={toList} onRetry={reload} />;
  const [subject, rec] = data;
  if (!subject || !rec) return <NotFound />;

  const missingKeys = Object.keys(rec.missing) as MissingKey[];
  const partial = missingKeys.length > 0;
  const base = `/subjects/${subject.id}`;

  return (
    <div className="screen" style={{ paddingBottom: partial ? 150 : 128 }}>
      <TopBar
        title="분석 결과지"
        nav="back"
        onNav={toList}
        subject={subject}
        subjectDetail={`${fmtDate(rec.measuredAt)} 측정`}
      />

      {partial && (
        <div className="alert alert-warn" role="status" style={{ margin: '20px 20px 0' }}>
          <TriangleAlertIcon size={22} />
          <div>
            <p className="title">일부 결과가 누락되었습니다</p>
            <p style={{ marginTop: 4, color: 'var(--fg-80)' }}>
              {missingKeys.map((k, i) => (
                <span key={k}>
                  {i > 0 && (i === missingKeys.length - 1 ? '와 ' : ', ')}
                  <b>{MISSING_LABEL[k]}</b>
                </span>
              ))}
              를 산출하지 못했습니다. 나머지 결과는 그대로 확인할 수 있습니다.
            </p>
          </div>
        </div>
      )}

      {/* 1. 대상자 정보 — 촬영 당시 값 */}
      <section className="card" style={{ margin: partial ? '16px 20px 0' : '20px 20px 0', padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
          <div style={{minWidth:0,overflowWrap:'anywhere'}}>
            <p style={{ fontSize: 14, fontWeight: 600 }} className="muted">대상자 이름</p>
            <h2 style={{ fontSize: 26, fontWeight: 800, lineHeight: 1.2 }}>{subject.name}</h2>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <p style={{ fontSize: 14, fontWeight: 600 }} className="muted">측정 날짜</p>
            <p style={{ fontSize: 17, fontWeight: 700 }}>{fmtDate(rec.measuredAt)}{rec.measuredAt && ` (${fmtWeekday(rec.measuredAt)})`}</p>
            <p style={{ fontSize: 13 }} className="muted">{!rec.live && fmtTime(rec.measuredAt)}</p>
          </div>
        </div>
        <dl className="metric-grid3" style={{ marginTop: 16, background: '#fff', border: '1px solid var(--border)', textAlign: 'center' }}>
          <div style={{ padding: '12px 4px' }}>
            <dt className="k">{rec.live ? "분석 구간" : "성별"}</dt>
            <dd style={{ fontSize: 18, fontWeight: 700 }}>{rec.live ? rec.live.analysis.segments.length+"개" : subject.sex}</dd>
          </div>
          <div style={{ padding: '12px 4px' }}>
            <dt className="k">촬영 당시 나이</dt>
            <dd style={{ fontSize: 18, fontWeight: 700 }}>{rec.ageAtCapture === null ? "미확인" : "만 "+rec.ageAtCapture+"세"}</dd>
          </div>
          <div style={{ padding: '12px 4px' }}>
            <dt className="k">촬영 당시 키</dt>
            <dd style={{ fontSize: 18, fontWeight: 700 }}>
              {rec.heightAtCapture ?? "미확인"}
              <span style={{ fontSize: 14, marginLeft: 2 }}>{rec.heightAtCapture !== null && "cm"}</span>
            </dd>
          </div>
        </dl>
      </section>

      {rec.live ? <>
        <div className="tabs" role="tablist" style={{margin:'20px 20px 0',gridTemplateColumns:'repeat(2,minmax(0,1fr))'}}>
          <button role="tab" aria-selected={!history} onClick={()=>setHistory(false)}>이번 결과</button>
          <button role="tab" aria-selected={history} onClick={()=>setHistory(true)}>지난 기록 비교</button>
        </div>
        {history ? <VisitHistory record={rec}/> : <LiveMetrics record={rec} selected={selected} select={i=>{setSelected(i);sessionStorage.setItem('segment-'+recordId,String(i));}} zoom={()=>navigate(base+'/results/'+rec.id+'/chart/knee')} openHistory={()=>{setHistory(true);window.scrollTo({top:0,behavior:'smooth'});}}/>}
      </> : <>
      <section style={{ margin: '28px 20px 0' }}>
        <h3 style={{ fontSize: 19, fontWeight: 800, marginBottom: 12 }}>핵심 지표</h3>
        <div style={{ borderRadius: 'var(--r-lg)', background: 'var(--primary)', color: '#fff', padding: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <p style={{ fontSize: 16, fontWeight: 700 }}>
              GDI <span style={{ fontWeight: 600, color: 'rgb(255 255 255 / .8)' }}>· 보행 편차 지수</span>
            </p>
            <button
              aria-label="GDI 설명"
              aria-expanded={Boolean(openInfo.gdi)}
              onClick={() => setOpenInfo((p) => ({ ...p, gdi: !p.gdi }))}
              style={{ width: 36, height: 36, borderRadius: 999, background: 'rgb(255 255 255 / .15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <InfoIcon size={18} />
            </button>
          </div>
          <p style={{ marginTop: 8, fontSize: 52, fontWeight: 800, lineHeight: 1, letterSpacing: '-0.02em' }}>{rec.gdi ?? '—'}</p>
          <p style={{ marginTop: 8, fontSize: 14, color: 'rgb(255 255 255 / .8)' }}>Gait Deviation Index</p>
          {openInfo.gdi && (
            <div style={{ marginTop: 16, borderRadius: 'var(--r-md)', background: 'rgb(255 255 255 / .15)', padding: '12px 16px', fontSize: 14, lineHeight: 1.6 }}>
              GDI 정의 및 해석 안내 문구는 확정 후 이곳에 표시됩니다.
            </div>
          )}
        </div>

        <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div className="card" style={{ padding: 16 }}>
            <p style={{ fontSize: 15, fontWeight: 700 }}>분당 걸음 수</p>
            <p style={{ fontSize: 13 }} className="muted">케이던스</p>
            <p style={{ marginTop: 12, fontSize: 34, fontWeight: 800, lineHeight: 1 }}>{rec.cadence ?? '—'}</p>
            <p style={{ marginTop: 6, fontSize: 14, fontWeight: 600, color: 'var(--fg-70)' }}>
              걸음/분 <span style={{ fontWeight: 400, color: 'var(--fg-50)' }}>(steps/min)</span>
            </p>
          </div>
          {rec.symmetry !== null ? (
            <div className="card" style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <p style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.35 }}>좌우 대칭<br />지표</p>
                <button
                  aria-label="계산 방식 보기"
                  aria-expanded={Boolean(openInfo.sym)}
                  className="info-btn"
                  onClick={() => setOpenInfo((p) => ({ ...p, sym: !p.sym }))}
                >
                  <InfoIcon size={18} />
                </button>
              </div>
              <p style={{ marginTop: 12, fontSize: 34, fontWeight: 800, lineHeight: 1 }}>{rec.symmetry}</p>
              <p style={{ marginTop: 6, fontSize: 14, fontWeight: 600, color: 'var(--fg-50)' }}>단위 확정 예정</p>
            </div>
          ) : (
            <div style={{ borderRadius: 'var(--r-lg)', border: '2px dashed rgb(247 185 40 / .6)', padding: 16 }}>
              <p style={{ fontSize: 15, fontWeight: 700, lineHeight: 1.35 }}>좌우 대칭<br />지표</p>
              <p style={{ marginTop: 12, fontSize: 34, fontWeight: 800, lineHeight: 1, color: 'var(--fg-30)' }}>—</p>
              <p style={{ marginTop: 6, fontSize: 14, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                <TriangleAlertIcon size={15} className="warn-icon" /> 산출되지 않음
              </p>
            </div>
          )}
        </div>
        {openInfo.sym && rec.symmetry !== null && (
          <div style={{ marginTop: 12, borderRadius: 'var(--r-md)', border: '2px dashed rgb(30 157 241 / .4)', background: 'rgb(227 236 246 / .5)', padding: '12px 16px' }}>
            <p style={{ fontSize: 14, fontWeight: 700 }} className="primary">좌우 대칭 지표 계산 방식</p>
            <p style={{ marginTop: 4, fontSize: 14, lineHeight: 1.6, color: 'var(--fg-80)' }}>계산식과 단위는 확정 후 이 영역에 표시됩니다.</p>
          </div>
        )}
        {rec.missing.symmetry && (
          <p style={{ marginTop: 12, fontSize: 14, lineHeight: 1.6, color: 'var(--fg-70)', display: 'flex', gap: 6 }}>
            <InfoIcon size={16} className="faint" /> 좌우 대칭 지표: {rec.missing.symmetry}
          </p>
        )}
      </section>

      {/* 3. 관절별 그래프 */}
      <section style={{ marginTop: 32 }}>
        <div style={{ margin: '0 20px' }}>
          <h3 style={{ fontSize: 19, fontWeight: 800 }}>관절별 그래프</h3>
          <div style={{ marginTop: 12 }}><LegendLR /></div>
          <AxisNote />
        </div>
        <div style={{ margin: '16px 20px 0', display: 'grid', gap: 16 }}>
          {JOINTS.map((j) => {
            const curve = rec.curves[j];
            return (
              <article key={j} style={{ borderRadius: 'var(--r-lg)', border: '1px solid var(--border)', padding: curve ? '16px 8px 8px' : 16 }}>
                <div style={{ padding: curve ? '0 12px' : 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <h4 style={{ fontSize: 17, fontWeight: 800 }}>{JOINT_LABEL[j]}</h4>
                  {curve ? (
                    <button className="zoom-btn" onClick={() => navigate(`${base}/results/${rec.id}/chart/${j}`)}>
                      <ExpandIcon size={15} /> 확대
                    </button>
                  ) : (
                    <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--fg-50)' }}>확대 불가</span>
                  )}
                </div>
                {curve ? (
                  <div style={{ marginTop: 4 }}>
                    <JointChart curve={curve} height={220} label={JOINT_LABEL[j]} />
                  </div>
                ) : (
                  <div
                    style={{ marginTop: 12, height: 180, borderRadius: 'var(--r-md)', border: '2px dashed rgb(247 185 40 / .6)', background: 'var(--card)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '0 24px' }}
                  >
                    <ChartLineIcon size={30} className="faint" />
                    <p style={{ marginTop: 12, fontSize: 16, fontWeight: 700 }}>{JOINT_LABEL[j]} 그래프를 표시할 수 없습니다</p>
                    <p style={{ marginTop: 4, fontSize: 14, lineHeight: 1.6 }} className="muted">{rec.missing[j] ?? '데이터를 산출하지 못했습니다.'}</p>
                  </div>
                )}
              </article>
            );
          })}
        </div>
        <p style={{ margin: '16px 20px 0', fontSize: 13, textAlign: 'center', color: 'var(--fg-50)' }}>
          결과 ID {rec.id} · 예시용 가상 데이터
        </p>
      </section>

      </>}
      <div className="bottom-cta" style={partial ? { paddingTop: 12 } : undefined}>
        {partial && <p className="hint sm">누락된 항목은 ‘—’로 표시되어 저장됩니다</p>}
        <button className="btn btn-primary" onClick={() => setDownloading(true)}>
          <DownloadIcon /> 결과 다운로드
        </button>
      </div>

      {downloading && <DownloadSheet record={rec} name={subject.name} onClose={() => setDownloading(false)} />}
    </div>
  );
}
