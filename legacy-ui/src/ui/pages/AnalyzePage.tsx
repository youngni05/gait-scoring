/** Local pose extraction followed by an idempotent measurement save. No AI job is implied. */
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { prepareRecording } from '../../../../mobile/src/pose/prepare-recording';
import type { PreparedMeasurement } from '../../../../contracts/measurement-1/prepare';
import { prepareAnalysisMeasurement } from '../../../../contracts/measurement-2/prepare';
import { ApiError, errorMessage } from '../../../../mobile/src/api/client';
import { apiSource, dataSource } from '../data';
import { PageError, PageLoading, TopBar } from '../components/layout';
import { useAsync } from '../session';
import { useLeaveGuard, useNavigate, useParams } from '../router';
import { NotFound } from './NotFound';
const Synthetic = import.meta.env.DEV && import.meta.env.VITE_E2E_MODE === 'true'
  ? lazy(() => import('../../e2e/SyntheticMeasurement')) : null;

export function AnalyzePage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const loaded = useAsync(signal => dataSource.getSubject(subjectId, signal), [subjectId]);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [message, setMessage] = useState('영상에서 관절 좌표를 추출한 뒤 서버에 저장합니다. 원본 영상은 전송하지 않습니다.');
  const [busy, setBusy] = useState(false);
  const [prepared, setPrepared] = useState<PreparedMeasurement | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [blocked, setBlocked] = useState(false);
  const [retryAt, setRetryAt] = useState(0);
  const [waiting, setWaiting] = useState(false);
  const active = useRef<AbortController | null>(null);
  const mount = useRef<HTMLDivElement>(null);
  useEffect(() => () => active.current?.abort(), []);
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    if (!retryAt) return;
    setWaiting(true);
    const timer = setTimeout(() => setWaiting(false), Math.max(0, retryAt - Date.now()));
    return () => clearTimeout(timer);
  }, [retryAt]);
  useLeaveGuard(busy || (!!prepared && !saved),
    '처리 중이거나 저장 상태를 확인하지 못한 요청이 있습니다. 화면을 떠나면 재시도 요청이 사라집니다. 나갈까요?');
  const back = () => navigate(`/subjects/${subjectId}`);
  async function prepare() {
    if (!file || !loaded.data || !mount.current || active.current) return;
    if (!loaded.data.birth || !loaded.data.heightCm) { setMessage('대상자의 생년월일과 키를 먼저 등록하세요.'); return; }
    const controller = new AbortController(); active.current = controller; setBusy(true); setBlocked(false);
    const analysisStartedAt = new Date().toISOString();
    setMessage('기기에서 영상의 전체 프레임을 읽고 관절 좌표를 추출하고 있습니다. 영상 길이에 따라 시간이 걸립니다.');
    try {
      const payload = await prepareRecording(file, mount.current, { subjectId, heightCm: loaded.data.heightCm, analysisStartedAt,
        recordingId: crypto.randomUUID(), requestId: crypto.randomUUID() }, controller.signal);
      if (controller.signal.aborted) return;
      setPrepared(prepareAnalysisMeasurement(payload, subjectId, analysisStartedAt));
      setMessage('좌표 준비 완료. 서버 저장 버튼을 눌러 측정을 저장하세요.');
    } catch (e) { if (!controller.signal.aborted) setMessage(errorMessage(e)); }
    finally { if (active.current === controller) { active.current = null; setBusy(false); } }
  }
  async function save() {
    if (!prepared || active.current || saved || blocked || Date.now() < retryAt) return;
    const controller = new AbortController(); active.current = controller; setBusy(true);
    setMessage('측정을 서버에 저장하고 있습니다.');
    try {
      const receipt = await apiSource.saveMeasurement(prepared, controller.signal);
      if (controller.signal.aborted) return;
      setSaved(receipt.id); setMessage('측정 저장 완료. 분석 결과는 아직 없습니다. 자동 AI 분석은 연결되지 않았습니다.');
    } catch (e) {
      setMessage(errorMessage(e));
      if (e instanceof ApiError) {
        setBlocked(e.status === 422 || e.code === 'IDEMPOTENCY_CONFLICT');
        if (e.retryAfter) setRetryAt(Date.now() + Math.min(e.retryAfter, 300) * 1000);
      }
    } finally { if (active.current === controller) { active.current = null; setBusy(false); } }
  }
  if (loaded.loading) return <PageLoading title="새 보행 측정" nav="back" onNav={back} />;
  if (loaded.error) return <PageError title="새 보행 측정" onNav={back} onRetry={loaded.reload} />;
  if (!loaded.data) return <NotFound />;
  return <div className="screen" style={{ paddingBottom: 40 }}>
    <TopBar title="새 보행 측정" nav="back" onNav={back} subject={loaded.data} />
    <section style={{ padding: 20, display: 'grid', gap: 20 }}>
      <label className="field">보행 영상 선택<input type="file" accept="video/*" disabled={busy || !!prepared} onChange={e => setFile(e.target.files?.[0] ?? null)} /></label>
      {preview && <video controls playsInline src={preview} style={{ width: '100%', maxHeight: 360, background: '#17232e', borderRadius: 16 }} />}
      <div ref={mount} hidden />
      <p role="status" aria-live="polite">{message}</p>
      {!prepared && <button className="btn btn-primary" disabled={!file || busy} onClick={() => void prepare()}>{busy ? '좌표 추출 중…' : '관절 좌표 추출'}</button>}
      {prepared && !saved && <button className="btn btn-primary" disabled={busy || blocked || waiting} onClick={() => void save()}>{waiting ? '잠시 후 재시도하세요' : busy ? '저장 중…' : '서버 저장 / 같은 요청 재시도'}</button>}
      {busy && <button className="btn btn-outline" onClick={() => { active.current?.abort(); setMessage(prepared ? '응답 대기를 취소했습니다. 저장되었을 수 있으니 같은 요청으로 재시도하세요.' : '좌표 추출을 취소했습니다.'); }}>취소</button>}
      {saved && <button className="btn btn-primary" onClick={() => navigate(`/subjects/${subjectId}/results/${saved}`)}>저장 기록 보기</button>}
      <button className="btn btn-outline" onClick={() => navigate(`/subjects/${subjectId}/results`)}>측정 기록 확인</button>
      {Synthetic && <Suspense fallback={<p>테스트 준비 중…</p>}><Synthetic participantId={subjectId} onPrepared={request => setPrepared(prepareAnalysisMeasurement(request, subjectId, new Date().toISOString()))} /></Suspense>}
    </section>
  </div>;
}
