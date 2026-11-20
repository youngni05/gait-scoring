/** Local pose extraction followed by an idempotent measurement save. No AI job is implied. */
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { prepareRecording } from '../../pose/prepare-recording';
import type { PreparedMeasurement } from '../../../../contracts/measurement-1/prepare';
import { prepareAnalysisMeasurement } from '../../../../contracts/measurement-2/prepare';
import { ApiError, errorMessage } from '../../api/client';
import { apiSource, dataSource } from '../data';
import { PageError, PageLoading, Steps, TopBar } from '../components/layout';
import { CheckCircleIcon, CheckIcon, FilmIcon, FolderOpenIcon, PlayIcon, RefreshIcon, SpinnerIcon, TriangleAlertIcon } from '../components/icons';
import { useAsync } from '../session';
import { useNavigate, useParams } from '../router';
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
  const [failed, setFailed] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);
  const [videoError, setVideoError] = useState(false);
  const [frames, setFrames] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
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
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => { if (busy || (prepared && !saved)) { e.preventDefault(); e.returnValue = ''; } };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [busy, prepared, saved]);
  const back = () => {
    if ((busy || (prepared && !saved)) && !window.confirm('저장 상태를 확인하지 못한 요청이 있습니다. 화면을 떠나면 재시도 요청이 사라집니다. 나갈까요?')) return;
    navigate(`/subjects/${subjectId}`);
  };
  async function prepare() {
    if (!file || videoError || !loaded.data || !mount.current || active.current) return;
    if (!loaded.data.birth || !loaded.data.heightCm) { setMessage('대상자의 생년월일과 키를 먼저 등록하세요.'); setFailed(true); return; }
    const controller = new AbortController(); active.current = controller; setBusy(true); setBlocked(false); setFailed(false); setFrames(0);
    const analysisStartedAt = new Date().toISOString();
    setMessage('기기에서 영상의 전체 프레임을 읽고 관절 좌표를 추출하고 있습니다. 영상 길이에 따라 시간이 걸립니다.');
    try {
      const payload = await prepareRecording(file, mount.current, { subjectId, heightCm: loaded.data.heightCm, analysisStartedAt,
        recordingId: crypto.randomUUID(), requestId: crypto.randomUUID() }, controller.signal, count => { if (!controller.signal.aborted) setFrames(count); });
      if (controller.signal.aborted) return;
      const request = prepareAnalysisMeasurement(payload, subjectId, analysisStartedAt);
      setPrepared(request);
      await saveRequest(request, controller);
    } catch (e) { if (!controller.signal.aborted) { setMessage(errorMessage(e)); setFailed(true); } }
    finally { if (active.current === controller) { active.current = null; setBusy(false); } }
  }
  async function save() {
    if (!prepared || active.current || saved || blocked || Date.now() < retryAt) return;
    const controller = new AbortController(); active.current = controller; setBusy(true); setFailed(false);
    try { await saveRequest(prepared, controller); }
    finally { if (active.current === controller) { active.current = null; setBusy(false); } }
  }
  async function saveRequest(request: PreparedMeasurement, controller: AbortController) {
    setMessage('측정을 서버에 저장하고 있습니다.');
    try {
      const receipt = await apiSource.saveMeasurement(request, controller.signal);
      if (controller.signal.aborted) return;
      setSaved(receipt.id); setMessage('측정 저장 완료. 분석 결과는 아직 없습니다. 자동 AI 분석은 연결되지 않았습니다.');
    } catch (e) {
      if (controller.signal.aborted) return;
      setMessage(errorMessage(e)); setFailed(true);
      if (e instanceof ApiError) {
        setBlocked(e.status === 422 || e.code === 'IDEMPOTENCY_CONFLICT');
        if (e.retryAfter) setRetryAt(Date.now() + Math.min(e.retryAfter, 300) * 1000);
      }
    }
  }
  if (loaded.loading) return <PageLoading title="새 보행 측정" nav="back" onNav={back} />;
  if (loaded.error) return <PageError title="새 보행 측정" onNav={back} onRetry={loaded.reload} />;
  if (!loaded.data) return <NotFound />;
  const openPicker = () => fileInput.current?.click();
  const inPreview = !!file && !busy && !prepared && !saved && !failed;
  const showPicker = !file && !prepared;
  const history = () => {
    if ((busy || (prepared && !saved)) && !window.confirm('저장 여부가 확인되지 않았습니다. 같은 요청으로 재시도할 수 없게 됩니다. 기록 목록으로 이동할까요?')) return;
    navigate(`/subjects/${subjectId}/results`);
  };
  return <div className="screen has-cta measurement-screen">
    <TopBar title="새 보행 영상 분석" nav="back" onNav={back} subject={loaded.data} />
    <Steps states={saved ? ['done', 'done', 'now'] : busy ? ['done', 'now', 'todo'] : failed ? ['done', 'fail', 'todo'] : ['now', 'todo', 'todo']} />
    <input ref={fileInput} type="file" accept="video/*" aria-label="보행 영상 선택" hidden disabled={busy || !!prepared}
      onChange={e => {
        const selected = e.target.files?.[0]; e.target.value = '';
        if (!selected) return;
        setFile(selected); setDuration(null); setVideoError(false); setFailed(false); setFrames(0);
        setMessage('');
      }} />
    <div ref={mount} hidden />
    {showPicker && <section className="measurement-section">
      <h2>분석할 보행 영상을 선택해 주세요</h2>
      <p className="muted">기기에 저장된 영상 중 1개를 선택합니다.</p>
      <button className="measurement-picker" onClick={openPicker}>
        <span className="measurement-film"><FilmIcon size={32} /></span>
        <strong>기기에서 영상 선택</strong>
        <span className="muted">갤러리 또는 파일에서 불러옵니다</span>
        <span className="btn btn-primary btn-sm"><FolderOpenIcon size={18} /> 영상 선택</span>
      </button>
    </section>}
    {inPreview && <section className="measurement-section">
      <div className="measurement-row"><h2>선택한 영상</h2><span className="primary"><CheckCircleIcon size={18} /> 선택 완료</span></div>
      {preview && <video key={preview} aria-label="선택한 보행 영상 미리보기" controls playsInline preload="metadata" src={preview} className="measurement-video"
        onLoadedMetadata={e => { const value = e.currentTarget.duration; setDuration(Number.isFinite(value) ? value : null); }}
        onError={() => { setVideoError(true); setMessage('이 영상을 재생할 수 없습니다. 재생 가능한 MP4 또는 MOV 영상을 선택해 주세요.'); }} />}
      <dl className="measurement-file-info"><dt>파일명</dt><dd>{file.name}</dd><dt>영상 길이</dt><dd>{duration === null ? '확인 중' : `${Math.floor(duration / 60)}분 ${Math.floor(duration % 60)}초`}</dd></dl>
      {videoError && <p role="alert" className="field-error">{message}</p>}
      <button className="btn btn-outline btn-sm" onClick={openPicker}><RefreshIcon size={18} /> 다른 영상 선택</button>
    </section>}
    {(busy || failed || saved || prepared) && <section className="measurement-section">
      <div className="card measurement-file-card"><FilmIcon size={28} /><strong>{file?.name ?? '테스트 좌표'}</strong></div>
      {busy && <div role="status" aria-live="polite" className="measurement-progress">
        <h2><SpinnerIcon size={22} /> {prepared ? '측정 기록을 저장하고 있습니다' : '영상을 처리하고 있습니다'}</h2>
        <p className="muted">{message}</p>
        {!prepared && <p className="measurement-frame-count">{frames.toLocaleString()}개 프레임 처리</p>}
        <ol className="measurement-stages">
          <li><CheckCircleIcon size={22} /> 영상 선택 완료</li>
          <li>{prepared ? <CheckCircleIcon size={22} /> : <SpinnerIcon size={22} />} 관절 좌표 추출 {prepared ? '완료' : '중'}</li>
          <li>{prepared ? <SpinnerIcon size={22} /> : <span className="measurement-dot" />} 측정 기록 저장 {prepared ? '중' : '대기'}</li>
        </ol>
      </div>}
      {failed && !busy && <div role="alert" className="measurement-failure">
        <span className="measurement-status-icon"><TriangleAlertIcon size={36} /></span>
        <h2>{prepared ? '저장 결과를 확인하지 못했습니다' : '영상을 처리하지 못했습니다'}</h2>
        <p>아래 원인을 확인한 뒤 다시 시도해 주세요.</p>
        <div className="measurement-cause"><strong>원인</strong><p>{message}</p></div>
        {prepared ? <p>중복 저장을 막기 위해 같은 요청으로 다시 시도합니다. 저장 여부는 측정 기록에서도 확인할 수 있습니다.</p> : <p>전신이 보이고 여러 걸음을 이어서 걷는 장면인지 확인해 주세요.</p>}
      </div>}
      {saved && <div role="status" className="measurement-success">
        <span className="measurement-status-icon"><CheckIcon size={40} /></span>
        <h2>측정 기록이 저장되었습니다</h2>
        <p>{loaded.data.name} 님의 관절 좌표를 저장했습니다.</p>
        <p className="muted">AI 분석 결과 생성은 아직 연결되지 않았습니다. 저장 기록에서 결과 유무를 확인할 수 있습니다.</p>
      </div>}
    </section>}
    {Synthetic && !busy && !saved && <section className="measurement-section"><Suspense fallback={<p>테스트 준비 중…</p>}><Synthetic participantId={subjectId} onPrepared={request => { setPrepared(prepareAnalysisMeasurement(request, subjectId, new Date().toISOString())); setFailed(false); }} /></Suspense></section>}
    <div className="bottom-cta">
      {!busy && !saved && !prepared && <>
        {!file && <p className="hint">영상을 선택하면 분석을 시작할 수 있습니다</p>}
        <button className="btn btn-primary" disabled={!file || videoError} onClick={() => void prepare()}><PlayIcon size={18} /> {failed ? '다시 시도' : '분석 시작'}</button>
        {failed && <button className="btn btn-outline" onClick={openPicker}><FolderOpenIcon /> 다른 영상 선택</button>}
      </>}
      {!busy && prepared && !saved && <button className="btn btn-primary" disabled={blocked || waiting} onClick={() => void save()}>{waiting ? '잠시 후 재시도하세요' : file ? '같은 요청으로 다시 시도' : '서버 저장 / 같은 요청 재시도'}</button>}
      {busy && <><p className="hint">처리가 끝날 때까지 화면을 닫지 말아 주세요.</p><button className="btn btn-outline" onClick={() => { active.current?.abort(); setFailed(true); setMessage(prepared ? '응답 대기를 취소했습니다. 저장되었을 수 있으니 같은 요청으로 재시도하세요.' : '영상 처리를 취소했습니다. 다시 시도하거나 다른 영상을 선택하세요.'); }}>취소</button></>}
      {saved && <button className="btn btn-primary" onClick={() => navigate(`/subjects/${subjectId}/results/${saved}`)}>저장 기록 보기</button>}
      {(prepared || saved) && !busy && <button className="btn btn-outline" onClick={history}>측정 기록 확인</button>}
    </div>
  </div>;
}
