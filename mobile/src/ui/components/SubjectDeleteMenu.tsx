import { useEffect, useRef, useState } from 'react';
import { ApiError } from '../../api/client';
import { dataSource } from '../data';
import type { SubjectSummary } from '../data/source';
import { TrashIcon } from './icons';
import './subject-delete.css';

export function SubjectDeleteMenu({ subject, onDeleted }: {
  subject: SubjectSummary; onDeleted: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const item = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    item.current?.focus();
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  const cancel = () => { setConfirming(false); trigger.current?.focus(); };
  return <div className="subject-delete-menu" ref={root} onKeyDown={event => {
    if (event.key === 'Escape' && open) {
      setOpen(false); trigger.current?.focus();
    }
  }} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }}>
    <button type="button" ref={trigger} className="icon-btn" aria-label="대상자 더보기"
      aria-expanded={open} aria-controls="subject-delete-options" onClick={() => setOpen(!open)}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <circle cx="12" cy="5" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="12" cy="19" r="2" />
      </svg>
    </button>
    {open && <div id="subject-delete-options" className="subject-delete-options">
      <button type="button" ref={item} onClick={() => { setOpen(false); setConfirming(true); }}>
        <TrashIcon size={18} />대상자 삭제
      </button>
    </div>}
    {confirming && <DeleteSubjectDialog subject={subject} onCancel={cancel} onDeleted={onDeleted} />}
  </div>;
}

function DeleteSubjectDialog({ subject, onCancel, onDeleted }: {
  subject: SubjectSummary; onCancel: () => void; onDeleted: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const inFlight = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const el = dialog.current!;
    el.showModal();
    return () => el.close();
  }, []);
  async function confirm() {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true); setError('');
    try {
      await dataSource.deleteSubject(subject.id);
      onDeleted();
    } catch (err) {
      setError(err instanceof ApiError && err.status === 409
        ? '영상 저장 또는 AI 분석이 진행 중입니다. 완료된 후 다시 삭제해 주세요.'
        : err instanceof ApiError && err.status === 401
          ? '로그인이 만료되었습니다. 다시 로그인해 주세요.'
          : '삭제 완료를 확인하지 못했습니다. 연결 상태를 확인한 뒤 다시 시도해 주세요.');
      inFlight.current = false; setBusy(false);
    }
  }
  return <dialog ref={dialog} className="sheet record-delete-sheet subject-delete-dialog"
    aria-labelledby="delete-subject-title" aria-describedby="delete-subject-note"
    onCancel={event => { event.preventDefault(); if (!inFlight.current) onCancel(); }}>
    <div className="grip" />
    <h2 id="delete-subject-title">{subject.name} 님을 삭제할까요?</h2>
    <p id="delete-subject-note">대상자 기본 정보와 모든 보행 분석 기록이 영구 삭제되며 복구할 수 없습니다.</p>
    {error && <p role="alert" className="field-error">{error}</p>}
    <div className="record-delete-actions">
      <button type="button" className="btn btn-outline" autoFocus disabled={busy} onClick={onCancel}>취소</button>
      <button type="button" className="btn record-delete-confirm" disabled={busy} onClick={confirm}>
        {busy ? '삭제 중…' : '삭제하기'}
      </button>
    </div>
  </dialog>;
}
