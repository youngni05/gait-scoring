import { useEffect, useRef, useState } from 'react';
import { ApiClient, ApiError, errorMessage, type Participant } from '../../mobile/src/api/client';
import { koreaToday, participantChanges, validateParticipant, type ParticipantInput, type ParticipantErrors } from '../../mobile/src/api/participant-input';

export function ParticipantForm({ client, original, onSaved, onClose }: {
  client: ApiClient; original: Participant | null; onSaved: (person: Participant) => void; onClose: () => void;
}) {
  const [input, setInput] = useState<ParticipantInput>({ name: original?.name ?? '', gender: original?.gender ?? '', birth_date: original?.birth_date ?? '', height_cm: String(original?.height_cm ?? '') });
  const [errors, setErrors] = useState<ParticipantErrors>({});
  const [message, setMessage] = useState(''), [busy, setBusy] = useState(false), [uncertain, setUncertain] = useState(false);
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => { active.current?.abort(); }, []);
  function change(field: keyof ParticipantInput, value: string) {
    setInput(v => ({ ...v, [field]: value })); setErrors(v => ({ ...v, [field]: '' })); setMessage('');
  }
  async function submit() {
    if (active.current || uncertain) return;
    const checked = validateParticipant(input); setErrors(checked.errors);
    if (!checked.body) { setMessage('입력 항목을 확인하세요.'); return; }
    const body = original ? participantChanges(original, checked.body) : checked.body;
    if (!Object.keys(body).length) { setMessage('변경한 정보가 없습니다.'); return; }
    const abort = new AbortController(); active.current = abort; setBusy(true); setMessage('저장 중…');
    try {
      const saved = original ? await client.updateParticipant(original.id, body, abort.signal) : await client.createParticipant(checked.body, abort.signal);
      if (!abort.signal.aborted) onSaved(saved);
    } catch (e) {
      if (abort.signal.aborted) return;
      // Participant POST has no idempotency key. Never offer blind retries after
      // timeout, 5xx or an unrecognized success response: creation may have committed.
      const ambiguous = e instanceof ApiError && (e.status === 0 || e.status >= 500 || e.code === 'INVALID_RESPONSE');
      if (ambiguous) { setUncertain(true); setMessage('저장 결과를 확인하지 못했습니다. 이미 반영되었을 수 있습니다. 닫은 뒤 대상자 목록을 새로고침하고 내용을 확인하세요.'); }
      else setMessage(e instanceof ApiError && e.status === 422 ? '대상자 정보가 서버 검증을 통과하지 못했습니다. 입력값을 확인하세요.' : errorMessage(e));
    } finally { if (!abort.signal.aborted) { active.current = null; setBusy(false); } }
  }
  const prefix = original ? 'edit-person' : 'new-person';
  return <section className="card" aria-label={original ? '대상자 정보 수정' : '대상자 등록'}>
    <h2>{original ? '대상자 정보 수정' : '대상자 등록'}</h2>
    <p>이름·성별·생년월일·키를 모두 입력하세요.</p>
    {original && <p className="muted">기본 정보를 수정해도 기존 측정 기록과 촬영 당시 나이는 바뀌지 않습니다. 현재 준비 중인 측정의 키도 그대로 유지됩니다.</p>}
    <form noValidate onSubmit={e => { e.preventDefault(); void submit(); }}>
      <fieldset disabled={busy || uncertain} style={{ border: 0, padding: 0, margin: 0 }}>
        <label>이름<input autoComplete="off" value={input.name} onChange={e => change('name', e.target.value)} aria-invalid={!!errors.name} aria-describedby={prefix + '-name'} /></label>
        <small id={prefix + '-name'}>{errors.name}</small>
        <label htmlFor={prefix + '-gender-input'}>성별</label><select id={prefix + '-gender-input'} value={input.gender} onChange={e => change('gender', e.target.value)} aria-invalid={!!errors.gender} aria-describedby={prefix + '-gender'}><option value="">선택하세요</option><option value="M">남성</option><option value="F">여성</option></select>
        <small id={prefix + '-gender'}>{errors.gender}</small>
        <label>생년월일<input type="date" max={koreaToday()} value={input.birth_date} onChange={e => change('birth_date', e.target.value)} aria-invalid={!!errors.birth_date} aria-describedby={prefix + '-birth'} /></label>
        <small id={prefix + '-birth'}>{errors.birth_date}</small>
        <label>키 (cm)<input type="text" inputMode="decimal" value={input.height_cm} onChange={e => change('height_cm', e.target.value)} aria-invalid={!!errors.height_cm} aria-describedby={prefix + '-height'} /></label>
        <small id={prefix + '-height'}>{errors.height_cm}</small>
        {Number(input.height_cm) > 0 && (Number(input.height_cm) < 100 || Number(input.height_cm) > 220) && <p>기본 정보로 등록할 수 있지만 현재 영상 측정은 키 100~220cm 범위만 지원합니다.</p>}
        <button type="submit">{busy ? '저장 중…' : original ? '수정 저장' : '등록 후 선택'}</button>
      </fieldset>
      <p role="status">{message}</p><button type="button" className="secondary" disabled={busy} onClick={onClose}>{uncertain ? '닫고 목록 확인' : '취소'}</button>
    </form>
  </section>;
}
