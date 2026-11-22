import { useContext, useEffect, useRef, useState } from 'react';
import { prepareMeasurement, type PreparedMeasurement } from '../../contracts/measurement-1/prepare';
import type { UploadRequest } from '../../mobile/src/pose/upload-gateway';
import { ApiSession } from '../../mobile/src/api/Session';
import { ApiError, errorMessage } from '../../mobile/src/api/client';

export function MeasurementUpload({ request }: { request: UploadRequest }) {
  const session = useContext(ApiSession);
  const [prepared, setPrepared] = useState<PreparedMeasurement | null>(null);
  const [message, setMessage] = useState('');
  const [saved, setSaved] = useState(false), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false);
  const [wait, setWait] = useState(false);
  const active = useRef<AbortController | null>(null), generation = useRef(0), retryAt = useRef(0);
  useEffect(() => {
    generation.current++; active.current?.abort(); active.current = null;
    setSaved(false); setBusy(false); setBlocked(false); setWait(false); retryAt.current = 0;
    try { setPrepared(prepareMeasurement(request, request.coords.subject.id)); setMessage('저장 준비 완료. 원본 영상은 전송하지 않습니다.'); }
    catch (e) { setPrepared(null); setMessage(errorMessage(e)); }
    return () => { generation.current++; active.current?.abort(); };
  }, [request]);
  useEffect(() => { if (!wait) return; const timer = setTimeout(() => setWait(false), Math.max(0, retryAt.current - Date.now())); return () => clearTimeout(timer); }, [wait]);
  async function save() {
    if (!session || !prepared || active.current || saved || blocked || Date.now() < retryAt.current) return;
    const id = generation.current, controller = new AbortController(); active.current = controller;
    setBusy(true); setMessage('서버에 저장 중…');
    try {
      const receipt = await session.client.save(prepared, controller.signal);
      if (generation.current !== id) return;
      setSaved(true); setMessage(`저장 완료 · 측정 ID: ${receipt.id}`); session.onSaved();
    } catch (e) {
      if (generation.current !== id) return;
      setMessage(errorMessage(e));
      if (e instanceof ApiError) {
        setBlocked(e.code === 'IDEMPOTENCY_CONFLICT' || e.status === 422);
        retryAt.current = Date.now() + Math.min(e.retryAfter, 300) * 1000;
        setWait(e.retryAfter > 0);
      }
    } finally { if (generation.current === id) { active.current = null; setBusy(false); } }
  }
  if (!session) return null;
  return <section aria-label="서버 저장"><h3>측정 저장</h3><p role="status">{message}</p>
    <button disabled={!prepared || busy || saved || blocked || wait} onClick={() => void save()}>{saved ? '저장 완료' : wait ? '잠시 후 재시도하세요' : '서버 저장 / 같은 요청 재시도'}</button>
    <button className="secondary" disabled={!busy} onClick={() => active.current?.abort()}>응답 대기 취소</button>
    <p className="muted">네트워크 오류나 인증 만료 후에는 이 버튼으로 재시도하세요. 새 JSON을 만들거나 화면을 닫으면 요청 키가 바뀔 수 있습니다. 취소는 서버 저장을 되돌리지 않습니다.</p>
  </section>;
}
