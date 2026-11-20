import { useState } from 'react';
import { koreaToday } from '../../api/participant-input';

/** Inline calendar works in embedded browsers without native showPicker support. */
export function BirthDatePicker({ value, onSelect, onClose }: {
  value: string | null; onSelect: (date: string) => void; onClose: () => void;
}) {
  const today = koreaToday();
  const initial = value ?? today;
  const [year, setYear] = useState(Number(initial.slice(0, 4)));
  const [month, setMonth] = useState(Number(initial.slice(5, 7)));
  const currentYear = Number(today.slice(0, 4));
  const first = new Date(year, month - 1, 1).getDay();
  const days = new Date(year, month, 0).getDate();
  const iso = (day: number) => `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
  const move = (delta: number) => {
    const date = new Date(year, month - 1 + delta, 1);
    setYear(date.getFullYear()); setMonth(date.getMonth() + 1);
  };
  return <section id="birth-calendar" aria-label="생년월일 달력" className="birth-calendar"
    onKeyDown={e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose(); } }}>
    <div className="birth-calendar-toolbar">
      <button type="button" aria-label="이전 달" disabled={year === 1900 && month === 1} onClick={() => move(-1)}>‹</button>
      <select aria-label="생년월일 연도" value={year} onChange={e => setYear(Number(e.target.value))}>
        {Array.from({ length: currentYear - 1899 }, (_, i) => currentYear - i).map(y => <option key={y} value={y}>{y}년</option>)}
      </select>
      <select aria-label="생년월일 월" value={month} onChange={e => setMonth(Number(e.target.value))}>
        {Array.from({ length: 12 }, (_, i) => i + 1).map(m => <option key={m} value={m}>{m}월</option>)}
      </select>
      <button type="button" aria-label="다음 달" disabled={year === currentYear && month >= Number(today.slice(5, 7))} onClick={() => move(1)}>›</button>
    </div>
    <div className="birth-calendar-days">
      {['일', '월', '화', '수', '목', '금', '토'].map(d => <span key={d} aria-hidden="true">{d}</span>)}
      {Array.from({ length: first }, (_, i) => <span key={`blank-${i}`} />)}
      {Array.from({ length: days }, (_, i) => i + 1).map(day => <button key={day} type="button"
        aria-label={`${year}년 ${month}월 ${day}일`} aria-pressed={value === iso(day)}
        disabled={iso(day) > today} onClick={() => onSelect(iso(day))}>{day}</button>)}
    </div>
    <button className="btn btn-outline btn-sm" type="button" onClick={onClose}>달력 닫기</button>
  </section>;
}
