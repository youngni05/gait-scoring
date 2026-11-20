/** 03 대상자 상세: 정보 확인 후 ‘새 보행 측정’ 또는 ‘이전 결과 조회’로 갈라진다. */
import type { ReactNode } from 'react';
import { Avatar, PageError, PageLoading, TopBar } from '../components/layout';
import { ChevronRightIcon, HistoryIcon, PenIcon, VideoIcon } from '../components/icons';
import { ageOn, dataSource, fmtBirth, fmtDate } from '../data';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { NotFound } from './NotFound';

export function SubjectDetailPage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(
    (signal) => Promise.all([dataSource.getSubject(subjectId, signal), dataSource.listResults(subjectId, signal)]),
    [subjectId],
  );
  const back = () => navigate('/subjects');
  if (loading) return <PageLoading title="대상자 상세" nav="back" onNav={back} />;
  if (error || !data) return <PageError title="대상자 상세" onNav={back} onRetry={reload} />;
  const [subject, records] = data;
  if (!subject) return <NotFound />;

  const info: [string, ReactNode][] = [
    ['나이', `만 ${ageOn(subject.birth)}세`],
    ['성별', subject.sex ?? '—'],
    ['키', <>{subject.heightCm ?? '—'} <span style={{ fontSize: 15, fontWeight: 600 }}>cm</span></>],
    ['생년월일', <span style={{ fontSize: 17, lineHeight: '30px' }}>{fmtBirth(subject.birth)}</span>],
  ];
  const latest = records[0];

  return (
    <div className="screen" style={{ paddingBottom: 40 }}>
      <TopBar title="대상자 상세" nav="back" onNav={back} />

      <section className="card" style={{ margin: '20px 20px 0', padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Avatar name={subject.name} size="lg" />
          <div style={{ flex: 1, minWidth: 0 }}>
            <p style={{ fontSize: 14, fontWeight: 600 }} className="primary">대상자</p>
            <h2 style={{ fontSize: 26, fontWeight: 800, lineHeight: 1.2 }}>{subject.name}</h2>
          </div>
        </div>
        <dl style={{ marginTop: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {info.map(([k, v]) => (
            <div key={k} style={{ borderRadius: 'var(--r-md)', background: '#fff', border: '1px solid var(--border)', padding: '12px 16px' }}>
              <dt style={{ fontSize: 14 }} className="muted">{k}</dt>
              <dd style={{ marginTop: 2, fontSize: 20, fontWeight: 700 }}>{v}</dd>
            </div>
          ))}
        </dl>
        <button
          className="btn btn-outline btn-sm"
          style={{ marginTop: 16, background: '#fff' }}
          onClick={() => navigate(`/subjects/${subject.id}/edit`)}
        >
          <PenIcon size={18} /> 대상자 정보 수정
        </button>
      </section>

      <section style={{ margin: '28px 20px 0' }}>
        <h3 style={{ fontSize: 17, fontWeight: 700, marginBottom: 12 }}>무엇을 하시겠어요?</h3>
        <div style={{ display: 'grid', gap: 12 }}>
          <button
            className="btn-primary"
            style={{ width: '100%', textAlign: 'left', borderRadius: 'var(--r-lg)', padding: 20, display: 'flex', alignItems: 'center', gap: 16 }}
            onClick={() => navigate(`/subjects/${subject.id}/analyze`)}
          >
            <span style={{ width: 56, height: 56, flexShrink: 0, borderRadius: 'var(--r-md)', background: 'rgb(255 255 255 / .2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <VideoIcon size={28} />
            </span>
            <span style={{ flex: 1 }}>
              <span style={{ display: 'block', fontSize: 20, fontWeight: 800 }}>새 보행 측정</span>
              <span style={{ display: 'block', marginTop: 4, fontSize: 15, color: 'rgb(255 255 255 / .85)' }}>보행 영상의 좌표를 추출해 저장합니다</span>
            </span>
            <ChevronRightIcon size={20} />
          </button>
          <button
            className="history-btn"
            style={{ width: '100%', textAlign: 'left', borderRadius: 'var(--r-lg)', padding: 20, display: 'flex', alignItems: 'center', gap: 16, background: 'var(--accent)', border: '2px solid rgb(30 157 241 / .2)' }}
            onClick={() => navigate(`/subjects/${subject.id}/results`)}
          >
            <span style={{ width: 56, height: 56, flexShrink: 0, borderRadius: 'var(--r-md)', background: '#fff', color: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <HistoryIcon size={28} />
            </span>
            <span style={{ flex: 1 }}>
              <span style={{ display: 'block', fontSize: 20, fontWeight: 800 }}>이전 결과 조회</span>
              <span style={{ display: 'block', marginTop: 4, fontSize: 15, color: 'var(--fg-70)' }}>
                {latest ? (
                  <>측정 기록 <b className="primary">{records.length}건</b> · 최근 {fmtDate(latest.measuredAt)}</>
                ) : (
                  '아직 측정 기록이 없습니다'
                )}
              </span>
            </span>
            <ChevronRightIcon size={20} className="primary" />
          </button>
        </div>
      </section>
    </div>
  );
}
