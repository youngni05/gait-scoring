import { ParticipantForm } from './ParticipantForm';
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { ApiClient, errorMessage, type Participant, type MeasurementItem, type MeasurementResult } from './api/client';
import { ApiSession } from './api/Session';
import { VideoTrial } from './VideoTrial';
import { resolveRecordedAt } from './pose/recorded-at';

const SyntheticMeasurement = import.meta.env.DEV && import.meta.env.VITE_E2E_MODE === 'true'
  ? lazy(() => import('./e2e/SyntheticMeasurement')) : null;

function History({ client, participant, revision }: { client: ApiClient; participant: string; revision: number }) {
  const [offset, setOffset] = useState(0), [refresh, setRefresh] = useState(0);
  const [rows, setRows] = useState<MeasurementItem[]>([]), [result, setResult] = useState<MeasurementResult | null>(null);
  const [message, setMessage] = useState(''), [busy, setBusy] = useState(false);
  const detail = useRef<AbortController | null>(null);
  useEffect(() => {
    const abort = new AbortController(); detail.current?.abort(); setResult(null); setRows([]); setBusy(true); setMessage('기록을 불러오는 중…');
    client.measurements(participant, offset, abort.signal).then(items => { if (!abort.signal.aborted) { setRows(items); setMessage(items.length ? '' : '저장된 기록이 없습니다.'); } }).catch(e => { if (!abort.signal.aborted) setMessage(errorMessage(e)); }).finally(() => { if (!abort.signal.aborted) setBusy(false); });
    return () => { abort.abort(); detail.current?.abort(); };
  }, [client, participant, offset, revision, refresh]);
  async function show(id: string) {
    detail.current?.abort(); const abort = new AbortController(); detail.current = abort; setResult(null); setMessage('결과를 불러오는 중…');
    try { const value = await client.result(id, abort.signal); if (!abort.signal.aborted) { setResult(value); setMessage(''); } }
    catch (e) { if (!abort.signal.aborted) setMessage(errorMessage(e)); }
  }
  return <section className="card"><h2>측정 기록</h2><button className="secondary" disabled={busy} onClick={() => { setOffset(0); setRefresh(v => v + 1); }}>새로고침</button><p role="status">{message}</p>
    <ul>{rows.map(row => <li key={row.id}><button className="secondary" onClick={() => void show(row.id)}>{new Date(row.measured_at).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' })} · {row.has_analysis_result ? '결과 보기' : '저장 기록 보기'}</button></li>)}</ul>
    <div className="actions"><button disabled={busy || offset === 0} onClick={() => setOffset(v => Math.max(0, v - 20))}>이전</button><button disabled={busy || rows.length < 20} onClick={() => setOffset(v => v + 20)}>다음</button></div>
    {result && <article><h3>측정 상세</h3><p>측정 ID: {result.measurement.id}</p><p>촬영 당시 만 {result.measurement.age}세</p>
      {result.status === 'not_available' ? <p>측정은 저장되었습니다. 분석 결과는 아직 없습니다. 자동 분석 실행 여부를 뜻하지 않습니다.</p> : <><p>GDI: {result.analysis_result!.gdi_score ?? '값 없음'}</p><p>보폭: {result.analysis_result!.stride_length_cm ?? '값 없음'} cm · 케이던스: {result.analysis_result!.cadence_spm ?? '값 없음'} steps/min</p><p>모델 버전: {result.analysis_result!.model_version}</p></>}
    </article>}
  </section>;
}

function Recording({ participant }: { participant: Participant }) {
  const [height, setHeight] = useState(String(participant.height_cm ?? ''));
  const [day, setDay] = useState(''), [metadata, setMetadata] = useState<string | null>(null);
  const { recordedAt, recordedAtSource } = resolveRecordedAt(day, metadata);
  return <section className="card"><h2>{participant.name} · 영상 측정</h2>
    <p>영상은 기기에서 처리하며 좌표와 처리 정보만 서버로 전송합니다.</p>
    {!participant.birth_date && <p role="alert">생년월일이 없습니다. 위의 대상자 수정에서 등록해야 측정을 저장할 수 있습니다.</p>}
    <label>촬영 당시 키 (cm)<input type="number" min="100" max="220" step="0.1" value={height} onChange={e => setHeight(e.target.value)} /></label>
    <label>촬영일 (직접 입력 시 우선 적용)<input placeholder="YYYY-MM-DD" value={day} onChange={e => setDay(e.target.value)} /></label>
    <small>날짜만 입력하면 한국 시간 00:00로 기록합니다. 비워 두면 영상 메타데이터를 사용합니다.</small>
    <p>촬영 시각: {recordedAt || '영상 선택 또는 날짜 입력 필요'} · {recordedAtSource === 'metadata' ? '영상 메타데이터' : '직접 입력'}</p>
    <VideoTrial input={{ subjectId: participant.id, heightCm: Number(height), recordedAt, recordedAtSource }} onMetadata={setMetadata} onSelection={() => { setDay(''); setMetadata(null); }} />
    {SyntheticMeasurement && <Suspense fallback={<p>테스트 준비 중…</p>}><SyntheticMeasurement participantId={participant.id} /></Suspense>}
  </section>;
}

export function ConnectedApp() {
  const [base, setBase] = useState(import.meta.env.VITE_API_BASE_URL ?? '');
  const [username, setUsername] = useState(''), [password, setPassword] = useState('');
  const token = useRef(''), loginAbort = useRef<AbortController | null>(null), listAbort = useRef<AbortController | null>(null);
  const [client, setClient] = useState<ApiClient | null>(null);
  const [message, setMessage] = useState('서버 주소와 테스트 계정으로 로그인하세요.'), [busy, setBusy] = useState(false);
  const [participants, setParticipants] = useState<Participant[]>([]), [selected, setSelected] = useState('');
  const [revision, setRevision] = useState(0);
  const [editor, setEditor] = useState<{ original: Participant | null } | null>(null);
  useEffect(() => () => { loginAbort.current?.abort(); listAbort.current?.abort(); token.current = ''; }, []);
  async function load(api: ApiClient) {
    listAbort.current?.abort(); const abort = new AbortController(); listAbort.current = abort;
    try { const people = await api.participants(abort.signal); if (!abort.signal.aborted) { setParticipants(people); setMessage(people.length ? '대상자를 선택하세요.' : '등록된 대상자가 없습니다. 새 대상자를 등록하세요.'); } }
    catch (e) { if (!abort.signal.aborted) setMessage(errorMessage(e)); }
  }
  async function login() {
    if (loginAbort.current) return;
    const abort = new AbortController(); loginAbort.current = abort; setBusy(true); setMessage('로그인 중…');
    try {
      const api = client ?? new ApiClient(base, () => token.current);
      const access = await api.login(username, password, abort.signal);
      if (abort.signal.aborted) return;
      token.current = access; setPassword(''); setClient(api);
      if (!client) await load(api); else { setRevision(v => v + 1); setMessage('인증 갱신 완료. 기존 저장 요청을 다시 시도할 수 있습니다.'); }
    } catch (e) { if (!abort.signal.aborted) setMessage(errorMessage(e)); }
    finally { if (loginAbort.current === abort) { loginAbort.current = null; setBusy(false); setPassword(''); } }
  }
  function logout() {
    if (!window.confirm('화면의 영상·좌표·재시도 요청이 지워집니다. 저장 중이거나 응답을 받지 못했다면 먼저 같은 요청으로 재시도하세요. 로그아웃할까요?')) return;
    loginAbort.current?.abort(); loginAbort.current = null; listAbort.current?.abort(); token.current = '';
    setEditor(null); setClient(null); setParticipants([]); setSelected(''); setPassword(''); setBusy(false); setMessage('로그아웃했습니다.');
  }
  function openEditor(original: Participant | null) {
    if (!original && selected && !window.confirm('새 대상자를 등록하면 선택이 바뀌고 현재 영상·좌표·재시도 요청이 지워집니다. 현재 측정의 저장 상태를 확인했나요?')) return;
    listAbort.current?.abort(); setEditor({ original });
  }
  function savedParticipant(person: Participant) {
    listAbort.current?.abort();
    setParticipants(people => [...people.filter(p => p.id !== person.id), person]);
    setSelected(person.id); setEditor(null); setRevision(v => v + 1);
    setMessage('대상자 정보를 저장했습니다. 선택된 대상자로 측정할 수 있습니다.');
  }
  const participant = participants.find(p => p.id === selected);
  return <main><header><span className="eyebrow">GAIT LAB</span><span className="badge">서버 연동</span></header>
    <section className="intro"><h1>보행 측정</h1><p>대상자를 선택하고 영상에서 추출한 좌표를 저장합니다.</p>{import.meta.env.DEV && !client && <a href="/dev.html">서버 없이 영상 처리 시험</a>}</section>
    <section className="card"><h2>{client ? '로그인 상태 / 인증 갱신' : '로그인'}</h2>
      <form onSubmit={e => { e.preventDefault(); void login(); }}>
        <label>서버 주소<input type="url" required placeholder="https://api.example.com" value={base} disabled={!!client || busy} onChange={e => setBase(e.target.value)} /></label>
        <label>아이디<input required autoComplete="username" value={username} disabled={!!client || busy} onChange={e => setUsername(e.target.value)} /></label>
        <label>비밀번호<input type="password" required autoComplete="current-password" value={password} disabled={busy} onChange={e => setPassword(e.target.value)} /></label>
        <button disabled={busy}>{busy ? '로그인 중…' : client ? '같은 계정으로 인증 갱신' : '로그인'}</button>
      </form><p role="status">{message}</p><small>비밀번호와 로그인 토큰은 저장소에 보관하지 않습니다. 새로고침하면 다시 로그인해야 합니다.</small>
      {client && <button className="secondary" onClick={logout}>로그아웃</button>}
    </section>
    {client && <ApiSession.Provider value={{ client, onSaved: () => setRevision(v => v + 1) }}>
      <section className="card"><h2>대상자 선택</h2><button className="secondary" disabled={!!editor} onClick={() => void load(client)}>대상자 목록 새로고침</button>
        <label>대상자<select disabled={!!editor} value={selected} onChange={e => { if (!selected || window.confirm('대상자를 바꾸면 현재 영상·좌표·재시도 요청이 지워집니다. 저장 상태를 확인했나요?')) setSelected(e.target.value); }}><option value="">선택하세요</option>{participants.map(p => <option key={p.id} value={p.id}>{p.name} · {p.birth_date ?? '생년월일 미등록'} · {p.id.slice(0, 8)}</option>)}</select></label>
      </section>
      <section className="card"><h2>대상자 관리</h2><div className="actions">
        <button disabled={!!editor} onClick={() => openEditor(null)}>새 대상자 등록</button>
        <button className="secondary" disabled={!!editor || !participant} onClick={() => participant && openEditor(participant)}>선택한 대상자 수정</button>
      </div></section>
      {editor && <ParticipantForm key={editor.original?.id ?? 'new'} client={client} original={editor.original} onSaved={savedParticipant} onClose={() => { setEditor(null); void load(client); }} />}
      {participant && <><Recording key={participant.id} participant={participant} /><History key={participant.id + ':' + revision} client={client} participant={participant.id} revision={revision} /></>}
    </ApiSession.Provider>}
  </main>;
}
