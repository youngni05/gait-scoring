/** 07-2 그래프 확대 보기: 관절을 바꿔 보며, 드래그한 구간을 확대한다. */
import { useState } from 'react';
import { JointChart } from '../components/JointChart';
import { AxisNote, LegendLR, PageError, PageLoading, TopBar } from '../components/layout';
import { HandIcon, PhoneIcon, ResetIcon } from '../components/icons';
import { JOINTS, JOINT_LABEL, dataSource, fmtDate, type Joint } from '../data';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { NotFound } from './NotFound';

export function ChartZoomPage() {
  const { subjectId = '', recordId = '', joint } = useParams();
  const navigate = useNavigate();
  const [resetKey, setResetKey] = useState(0);
  const { data, loading, error, reload } = useAsync(
    (signal) => Promise.all([dataSource.getSubject(subjectId, signal), dataSource.getResult(subjectId, recordId, signal)]),
    [subjectId, recordId],
  );
  const reportUrl = `/subjects/${subjectId}/results/${recordId}`;
  const close = () => navigate(reportUrl);
  if (loading) return <PageLoading title="그래프 확대 보기" nav="close" onNav={close} />;
  if (error || !data) return <PageError title="그래프 확대 보기" onNav={close} onRetry={reload} />;
  const [subject, rec] = data;
  const j = joint as Joint;
  if (!subject || !rec || !JOINTS.includes(j)) return <NotFound />;
  const curve = rec.curves[j];

  return (
    <div className="screen" style={{ paddingBottom: 32 }}>
      <TopBar
        title="그래프 확대 보기"
        nav="close"
        onNav={close}
        subject={subject}
        subjectDetail={`${fmtDate(rec.measuredAt)} 측정`}
      />

      <div className="tabs" role="tablist" aria-label="관절 선택" style={{ margin: '16px 20px 0' }}>
        {JOINTS.map((k) => (
          <button
            key={k}
            role="tab"
            aria-selected={k === j}
            disabled={!rec.curves[k]}
            title={rec.curves[k] ? undefined : '산출되지 않은 그래프입니다'}
            onClick={() => navigate(`${reportUrl}/chart/${k}`, { replace: true })}
          >
            {JOINT_LABEL[k]}
          </button>
        ))}
      </div>

      <div style={{ margin: '16px 20px 0', display: 'flex', alignItems: 'center', gap: 20 }}>
        <LegendLR />
        <button
          onClick={() => setResetKey((n) => n + 1)}
          style={{ marginLeft: 'auto', height: 40, padding: '0 14px', borderRadius: 999, border: '2px solid var(--border)', fontSize: 14, fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 4 }}
        >
          <ResetIcon size={15} /> 원래대로
        </button>
      </div>

      <section style={{ margin: '12px 12px 0', borderRadius: 'var(--r-lg)', border: '1px solid var(--border)', padding: '12px 4px 4px' }}>
        <p style={{ padding: '0 16px', fontSize: 18, fontWeight: 800 }}>{JOINT_LABEL[j]}</p>
        {curve ? (
          <JointChart key={j} curve={curve} height={460} zoomable resetKey={resetKey} label={JOINT_LABEL[j]} />
        ) : (
          <p style={{ padding: 48, textAlign: 'center' }} className="muted">이 관절의 그래프는 산출되지 않았습니다.</p>
        )}
      </section>
      <div style={{ margin: '8px 20px 0' }}>
        <AxisNote />
      </div>

      <div className="card" style={{ margin: '16px 20px 0', borderRadius: 'var(--r-md)', padding: '14px 16px', display: 'grid', gap: 8, fontSize: 14, color: 'var(--fg-80)' }}>
        <p style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <HandIcon size={18} className="primary" /> 그래프를 가로로 드래그하면 해당 구간이 확대되고, 누르면 그 지점의 값이 보입니다.
        </p>
        <p style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <PhoneIcon size={18} className="primary" /> 휴대폰을 가로로 돌리면 더 넓게 볼 수 있습니다.
        </p>
      </div>
    </div>
  );
}
