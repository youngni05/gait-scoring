/**
 * 08 결과 다운로드 바닥 시트: 준비 중 → 완료 / 실패.
 * 실제 결과 API 응답을 JSON으로 내려받는다. 브라우저의 파일 저장 완료 여부는 단정하지 않는다.
 */
import { useEffect, useState } from 'react';
import { dataSource, fmtCompact, fmtDate, type ResultRecord } from '../data';
import { CheckCircleIcon, CheckIcon, CloseIcon, FileIcon, RotateIcon, SpinnerIcon } from './icons';

type Phase = 'preparing' | 'done' | 'fail';

export function DownloadSheet({ record, name, onClose }: { record: ResultRecord; name: string; onClose: () => void }) {
  const [phase, setPhase] = useState<Phase>('preparing');
  const [attempt, setAttempt] = useState(0);
  const [failure, setFailure] = useState('');
  const fileName = `gait-result-${record.id}.json`;

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    setPhase('preparing');
    dataSource
      .downloadResult(record.id, controller.signal)
      .then(() => alive && setPhase('done'))
      .catch((e) => { if (alive) { setFailure(e instanceof Error ? e.message : '결과를 내려받지 못했습니다.'); setPhase('fail'); } });
    return () => {
      alive = false;
      controller.abort();
    };
  }, [attempt, record.id]);

  // 준비 중에는 바깥을 눌러도 닫히지 않게 한다(작업이 끝나기 전에 사라지는 혼란 방지).
  const closable = phase !== 'preparing';

  return (
    <>
      <div className="sheet-backdrop" onClick={closable ? onClose : undefined} />
      <section
        className="sheet"
        role={phase === 'fail' ? 'alert' : 'status'}
        aria-live="polite"
        aria-label="결과 다운로드"
      >
        <div className="grip" />
        <div style={{ paddingTop: 28, textAlign: 'center' }}>
          {phase === 'preparing' && (
            <>
              <span className="round-icon" style={{ background: 'var(--accent)', color: 'var(--primary)' }}>
                <SpinnerIcon size={28} />
              </span>
              <h2 style={{ marginTop: 16, fontSize: 21, fontWeight: 800 }}>결과 파일을 준비하고 있습니다</h2>
              <p style={{ marginTop: 6, fontSize: 15 }} className="muted">잠시만 기다려 주세요.</p>
            </>
          )}
          {phase === 'done' && (
            <>
              <span className="round-icon" style={{ background: 'rgb(23 191 99 / .15)', color: 'var(--ok)' }}>
                <CheckIcon size={28} />
              </span>
              <h2 style={{ marginTop: 16, fontSize: 21, fontWeight: 800 }}>다운로드를 요청했습니다</h2>
              <p style={{ marginTop: 6, fontSize: 15 }} className="muted">브라우저의 다운로드 목록에서 저장 여부를 확인하세요.</p>
            </>
          )}
          {phase === 'fail' && (
            <>
              <span className="round-icon" style={{ background: 'rgb(244 33 46 / .1)', color: 'var(--danger)' }}>
                <CloseIcon size={28} />
              </span>
              <h2 style={{ marginTop: 16, fontSize: 21, fontWeight: 800 }}>다운로드하지 못했습니다</h2>
              <p style={{ marginTop: 6, fontSize: 15 }} className="muted">
                {name} · {fmtDate(record.measuredAt)} 측정 결과지
              </p>
            </>
          )}
        </div>

        {phase === 'fail' ? (
          <div
            style={{ marginTop: 24, borderRadius: 'var(--r-md)', border: '2px solid rgb(244 33 46 / .25)', background: 'rgb(244 33 46 / .05)', padding: 16 }}
          >
            <p style={{ fontSize: 14, fontWeight: 700, color: 'var(--danger)' }}>원인</p>
            <p style={{ marginTop: 4, fontSize: 16, fontWeight: 700, lineHeight: 1.4 }}>
              {failure}
            </p>
            <p style={{ marginTop: 8, fontSize: 14, lineHeight: 1.6, color: 'var(--fg-70)' }}>
              저장 공간과 인터넷 연결을 확인한 뒤 다시 시도해 주세요.
            </p>
          </div>
        ) : (
          <div className="card" style={{ marginTop: 24, borderRadius: 'var(--r-md)', padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <span
              style={{
                width: 44, height: 44, borderRadius: 'var(--r-md)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                background: phase === 'done' ? 'rgb(23 191 99 / .15)' : 'var(--accent)',
                color: phase === 'done' ? 'var(--ok)' : 'var(--primary)',
              }}
            >
              <FileIcon size={22} />
            </span>
            <div style={{ minWidth: 0, flex: 1 }}>
              <p style={{ fontSize: 15, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{fileName}</p>
              <p style={{ marginTop: 2, fontSize: 13, color: 'var(--fg-60)', display: 'flex', alignItems: 'center', gap: 6 }}>
                파일 형식 <span className="pill">JSON</span>
              </p>
            </div>
            {phase === 'done' && <CheckCircleIcon size={22} className="ok-icon" />}
          </div>
        )}

        {phase === 'preparing' && (
          <div className="bar indeterminate" style={{ marginTop: 20 }}>
            <div />
          </div>
        )}
        {phase === 'done' && (
          <button className="btn btn-primary" style={{ marginTop: 24 }} onClick={onClose}>
            확인
          </button>
        )}
        {phase === 'fail' && (
          <div style={{ marginTop: 24, display: 'grid', gap: 12 }}>
            <button className="btn btn-primary" onClick={() => setAttempt((n) => n + 1)}>
              <RotateIcon /> 다시 시도
            </button>
            <button className="btn btn-outline" onClick={onClose}>
              닫기
            </button>
          </div>
        )}
      </section>
    </>
  );
}
