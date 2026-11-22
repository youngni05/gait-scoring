/** 03 대상자 상세: 정보 확인 후 ‘새 보행 영상 분석’ 또는 ‘이전 결과 조회’로 갈라진다. */
import { Avatar, PageError, PageLoading, TopBar } from '../components/layout';
import { ChevronRightIcon, HistoryIcon, PenIcon, VideoIcon } from '../components/icons';
import { ageOn, dataSource, fmtBirth, fmtDate } from '../data';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { fmtDay, subjectOverview, type SubjectOverview } from '../data/visit-summary.mjs';
import { ChangeBadge } from '../components/ChangeBadge';
import { signed } from './VisitMetrics';
import { NotFound } from './NotFound';
import '../report.css';
import { SubjectDeleteMenu } from '../components/SubjectDeleteMenu';

export function SubjectDetailPage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(
    (signal) => dataSource.getSubjectSummary(subjectId, signal),
    [subjectId],
  );
  const back = () => navigate('/subjects');
  if (loading) return <PageLoading title="대상자 상세" nav="back" onNav={back} />;
  if (error || !data) return <PageError title="대상자 상세" onNav={back} onRetry={reload} />;
  const subject = data;
  if (!subject) return <NotFound />;

  const previews = subject.recentVisits ?? [];
  const overview = previews.length ? subjectOverview(previews) : null;

  return (
    <div className="screen" style={{ paddingBottom: 40 }}>
      <TopBar title="대상자 상세" nav="back" onNav={back}
        actions={<SubjectDeleteMenu subject={subject} onDeleted={back} />} />

      <section className="subject-identity" aria-label="대상자 정보">
        <div className="subject-identity-head">
          <Avatar name={subject.name} size="sm" />
          <h2>{subject.name}</h2>
          <button className="subject-edit" aria-label="대상자 정보 수정" onClick={() => navigate(`/subjects/${subject.id}/edit`)}>
            <PenIcon size={16} />수정
          </button>
        </div>
        <dl className="subject-facts">
          <div><dt>나이</dt><dd>{subject.birth ? `만 ${ageOn(subject.birth)}세` : '미확인'}</dd></div>
          <div><dt>성별</dt><dd>{subject.sex ?? '미확인'}</dd></div>
          <div><dt>키</dt><dd>{subject.heightCm == null ? '미확인' : `${subject.heightCm} cm`}</dd></div>
          <div><dt>생년월일</dt><dd>{fmtBirth(subject.birth) || '미확인'}</dd></div>
        </dl>
      </section>

      {overview?.latest && (
        <section className="subject-summary-section">
          <button className="ovw-summary" onClick={() => navigate(`/subjects/${subject.id}/overview`)}>
            <span className="t">보행 변화 요약<small>최근 {overview.visits.length}회 · 전체 보기 <ChevronRightIcon size={14} /></small></span>
            <ChangeBlocks overview={overview} />
          </button>
        </section>
      )}

      {!overview?.latest && subject.resultCount > 0 && <section className="subject-summary-section">
        <button className="ovw-summary" onClick={() => navigate(`/subjects/${subject.id}/overview`)}>
          <span className="t">보행 변화 요약<small>전체 기록 해석 보기 <ChevronRightIcon size={14} /></small></span>
          <span className="ovw-baseline">비교 가능한 기록 확인</span>
        </button>
      </section>}

      <section className="subject-action-section">
        <h3>무엇을 하시겠어요?</h3>
        <div className="subject-actions">
          <button
            className="btn-primary"
            onClick={() => navigate(`/subjects/${subject.id}/analyze`)}
          >
            <span className="subject-action-icon">
              <VideoIcon size={28} />
            </span>
            <span style={{ flex: 1 }}>
              <span className="subject-action-title">새 보행 영상 분석</span>
              <span className="subject-action-note">휴대폰의 보행 영상을 선택해 분석합니다</span>
            </span>
            <ChevronRightIcon size={20} />
          </button>
          <button
            className="history-btn"
            onClick={() => navigate(`/subjects/${subject.id}/results`)}
          >
            <span className="subject-action-icon">
              <HistoryIcon size={28} />
            </span>
            <span style={{ flex: 1 }}>
              <span className="subject-action-title">이전 결과 조회</span>
              <span className="subject-action-note muted">
                {subject.resultCount > 0 ? (
                  <>분석 기록 <b className="primary">{subject.resultCount}건</b> · 최근 {fmtDate(subject.latestMeasuredAt ?? '')}</>
                ) : (
                  '아직 분석 기록이 없습니다'
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

function ChangeBlocks({ overview }: { overview: SubjectOverview }) {
  const { trends, visits, history } = overview;
  const recent = history?.recent;
  if (!recent || !trends.length) {
    return <span className="ovw-state">{visits.length === 1 ? '첫 기록 · 다음 기록부터 비교' : '비교 가능한 지표 없음'}</span>;
  }

  // 최근 달라진 지표를 먼저, 그다음 여러 기록에서 비교할 수 있는 지표를 표시한다.
  const priority = (t: SubjectOverview['trends'][number]) =>
    t.verdict !== 'similar' ? 0 : t.points.length > 2 ? 1 : 2;
  const visible = [...trends].sort((a, b) => priority(a) - priority(b)).slice(0, 2);
  return <>
    <span className="ovw-baseline">최근 비교 · {fmtDay(recent.previousDate)} → {fmtDay(visits[0]!.date)}</span>
    <span className="ovw-blocks">
      {visible.map(t => {
        const points = t.points.slice(-4);
        return <span className="ovw-block" key={t.key}>
          <span className="ovw-metric">{t.label}</span>
          <strong>{signed(t.delta, t.digits)}<small>{t.key === 'cadence' ? '걸음/분' : t.unit}</small></strong>
          <ChangeBadge delta={t.delta} verdict={t.verdict} />
          {points.length >= 3 && <span className="ovw-series">
            <span>최근 {points.length}회 흐름</span>
            <b>{points.map(p => p.value.toFixed(t.digits)).join(' → ')}</b>
          </span>}
        </span>;
      })}
    </span>
  </>;
}
