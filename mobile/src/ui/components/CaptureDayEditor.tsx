import { useState } from 'react';
import { displayDay } from '../data/mock';
import { ApiError } from '../../api/client';
import { dataSource, type ResultRecord } from '../data';
import { CaptureDateField } from './CaptureDateField';
import { PenIcon } from './icons';

export function CaptureDayEditor({ record, birth, reload }: { record: ResultRecord; birth: string; reload: () => void }) {
  const current = record.captureDay === undefined
    ? (record.dateBasis === 'analysis_start' ? '' : displayDay(record.measuredAt))
    : record.captureDay ?? '';
  const [draft, setDraft] = useState<string | null>(null);
  const [previousDay, setPreviousDay] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function save() {
    if (!draft || busy || draft === current) return;
    setBusy(true);
    setError('');
    try {
      await dataSource.updateCaptureDay(record.subjectId, record.id, draft, previousDay);
      setDraft(null);
      reload();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setDraft(null);
        setError('다른 곳에서 촬영일을 변경했습니다. 최신 날짜를 확인해 주세요.');
        reload();
      } else {
        setError(e instanceof ApiError && e.status === 422 ? '생년월일부터 오늘까지의 날짜를 선택해 주세요.' : '촬영일을 저장하지 못했습니다. 다시 시도해 주세요.');
      }
    } finally { setBusy(false); }
  }
  return <div className="record-capture">
    <div className="record-capture-row">
      <div><span className="muted">촬영일</span><b>{current ? current.replaceAll('-', '.') : '미확인'}</b></div>
      {draft === null && <button type="button" className="capture-edit-action" onClick={() => {setError(''); setPreviousDay(current || null); setDraft(current);}}>
        <PenIcon size={16}/> 촬영일 수정
      </button>}
    </div>
    {draft !== null && <fieldset className="capture-fields" disabled={busy}>
      <CaptureDateField label="수정할 촬영일" value={draft} min={birth} onChange={setDraft} allowClear={false}/>
      <div className="capture-actions">
        <button className="btn btn-outline btn-sm" type="button" onClick={() => {setError(''); setDraft(null);}}>취소</button>
        <button className="btn btn-primary btn-sm" type="button" disabled={!draft || draft === current || busy} onClick={save}>{busy ? '저장 중…' : '변경 저장'}</button>
      </div>
    </fieldset>}
    {error && <p role="alert" className="field-error">{error}</p>}
  </div>;
}
