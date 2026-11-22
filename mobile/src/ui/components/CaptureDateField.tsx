import { useId, useRef, useState } from 'react';
import { CalendarIcon, ChevronRightIcon } from './icons';

export function CaptureDateField({ value, onChange, label = '촬영일', min = '', allowClear = true }: { value: string; onChange: (day: string) => void; label?: string; min?: string; allowClear?: boolean }) {
  const id = useId();
  const today = new Intl.DateTimeFormat('sv-SE', {timeZone: 'Asia/Seoul'}).format(new Date());
  const [month, setMonth] = useState<Date | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  function close() {
    setMonth(null);
    trigger.current?.focus();
  }
  function choose(day: string) {
    onChange(day);
    close();
  }
  function browse(offset: number) {
    if (month) setMonth(new Date(month.getFullYear(), month.getMonth() + offset, 1));
  }
  const prefix = month && `${month.getFullYear()}-${String(month.getMonth() + 1).padStart(2, '0')}-`;
  return <div className="field" onKeyDown={e => { if (month && e.key === 'Escape') { e.preventDefault(); close(); } }}>
    <label htmlFor={id}>{label}</label>
    <button id={id} type="button" ref={trigger} className="input capture-date-trigger" aria-expanded={!!month}
      onClick={() => {
        if (month) { close(); return; }
        const date = value ? new Date(value + 'T12:00:00') : new Date();
        setMonth(new Date(date.getFullYear(), date.getMonth(), 1));
      }}>
      {value ? value.replaceAll('-', '.') : '촬영일 선택'}<CalendarIcon size={20} />
    </button>
    {month && <div className="capture-calendar" role="group" aria-label="촬영일 달력">
      <div className="capture-month">
        <button type="button" aria-label="이전 연도" onClick={() => browse(-12)}>«</button>
        <button type="button" aria-label="이전 달" onClick={() => browse(-1)}><ChevronRightIcon size={16} className="reverse" /></button>
        <strong aria-live="polite">{month.getFullYear()}년 {month.getMonth() + 1}월</strong>
        <button type="button" aria-label="다음 달" onClick={() => browse(1)}><ChevronRightIcon size={16} /></button>
        <button type="button" aria-label="다음 연도" onClick={() => browse(12)}>»</button>
      </div>
      <div className="capture-days">
        {Array.from('일월화수목금토', key => <span key={key} aria-hidden="true">{key}</span>)}
        {Array.from({length: new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate()}, (_, i) => {
          const day = prefix + String(i + 1).padStart(2, '0');
          return <button key={day} type="button" style={i === 0 ? {gridColumnStart: month.getDay() + 1} : undefined}
            disabled={day < min || day > today} aria-label={day} aria-pressed={day === value} onClick={() => choose(day)}>{i + 1}</button>;
        })}
      </div>
      {allowClear && value && <button className="capture-clear" type="button" onClick={() => choose('')}>선택 해제</button>}
    </div>}
  </div>;
}
