/** 여러 화면이 함께 쓰는 상단 표시줄·현재 대상자 표시줄·진행 단계 표시. */
import type { ReactNode } from 'react';
import { ageOn, type Subject } from '../data';
import { LoadingMark } from './brand';
import { ArrowLeftIcon, CheckIcon, CloseIcon } from './icons';

export function Avatar({ name, size, tone }: { name: string; size: 'sm' | 'md' | 'lg'; tone?: 'a' | 'b' }) {
  return (
    <span className={`avatar ${size} ${tone ? `tone-${tone}` : ''}`} aria-hidden="true">
      {name.slice(0, 1)}
    </span>
  );
}

/**
 * 03 대상자 상세 이후 모든 화면에서 유지되는 ‘현재 대상자’ 표시줄.
 * 다른 대상자의 영상을 잘못 분석하는 실수를 막으려는 장치다.
 */
export function SubjectBar({ subject, detail }: { subject: Subject; detail?: string | undefined }) {
  const text = detail ?? [subject.sex,subject.birth && ageOn(subject.birth)+'세'].filter(Boolean).join(' · ');
  return (
    <div className="subject-bar">
      <div className="subject-bar-inner">
        <Avatar name={subject.name} size="sm" />
        <p>
          <span className="label">현재 대상자</span>
          <span className="sep">|</span>
          <b>{subject.name}</b> · {text}
        </p>
      </div>
    </div>
  );
}

interface TopBarProps {
  title: string;
  /** 'back'은 화살표, 'close'는 X. 생략하면 버튼 없이 제목만 둔다. */
  nav?: 'back' | 'close' | undefined;
  onNav?: (() => void) | undefined;
  subject?: Subject | null | undefined;
  subjectDetail?: string | undefined;
  sticky?: boolean;
  children?: ReactNode;
  actions?: ReactNode;
}

export function TopBar({ title, nav, onNav, subject, subjectDetail, sticky = true, children, actions }: TopBarProps) {
  return (
    <header className="topbar" style={sticky ? undefined : { position: 'static' }}>
      <div className={`topbar-row ${nav ? '' : 'no-back'}`}>
        {nav && (
          <button className="icon-btn" aria-label={nav === 'back' ? '뒤로' : '닫기'} onClick={onNav}>
            {nav === 'back' ? <ArrowLeftIcon size={22} /> : <CloseIcon size={22} />}
          </button>
        )}
        <h1>{title}</h1>
        {actions}
      </div>
      {subject && <SubjectBar subject={subject} detail={subjectDetail} />}
      {children}
    </header>
  );
}

export type StepState = 'todo' | 'now' | 'done' | 'fail';

/** 영상 선택 → 분석 → 결과지 세 단계 중 지금 위치를 보여준다. */
export function Steps({ states }: { states: [StepState, StepState, StepState] }) {
  const names = ['영상 선택', '분석', '결과지'];
  return (
    <ol className="steps" aria-label="진행 단계">
      {names.map((name, i) => {
        const st = states[i]!;
        const next = states[i + 1];
        return (
          <li key={name} style={{ display: 'contents' }}>
            <span className={`step ${st}`} aria-current={st === 'now' ? 'step' : undefined}>
              <span className="dot">
                {st === 'done' ? <CheckIcon size={13} /> : st === 'fail' ? <CloseIcon size={13} /> : i + 1}
              </span>
              <span className="name">{name}</span>
            </span>
            {next && (
              <span
                className={`line ${next === 'fail' ? 'fail' : st === 'done' && next !== 'todo' ? 'on' : ''}`}
                aria-hidden="true"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function LegendLR() {
  return (
    <div className="legend">
      <span>
        <svg width="28" height="8" aria-hidden="true">
          <line x1="0" y1="4" x2="28" y2="4" stroke="var(--primary)" strokeWidth="4" strokeLinecap="round" />
        </svg>
        왼쪽
      </span>
      <span>
        <svg width="28" height="8" aria-hidden="true">
          <line x1="0" y1="4" x2="28" y2="4" stroke="var(--right)" strokeWidth="4" strokeDasharray="7 4" />
        </svg>
        오른쪽
      </span>
    </div>
  );
}

/** 축 이름과 단위가 아직 확정되지 않았음을 알리는 표시. */
export function AxisNote() {
  return (
    <p style={{ marginTop: 8, fontSize: 13, color: 'var(--fg-60)', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
      <span className="pill pill-draft">가안</span>
      가로축 보행 주기(%) · 세로축 관절 각도(°)
    </p>
  );
}

/** 데이터를 불러오는 동안 보여주는 화면. */
export function PageLoading({ title, nav, onNav }: { title: string; nav?: 'back' | 'close'; onNav?: () => void }) {
  return (
    <div className="screen">
      <TopBar title={title} nav={nav} onNav={onNav} />
      <p role="status" className="primary" style={{ padding: '72px 20px 40px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, fontSize: 15, fontWeight: 600 }}>
        <LoadingMark size={44} /> 불러오는 중입니다…
      </p>
    </div>
  );
}

/** 불러오기에 실패했을 때 보여주는 화면. */
export function PageError({ title, onNav, onRetry }: { title: string; onNav: () => void; onRetry: () => void }) {
  return (
    <div className="screen">
      <TopBar title={title} nav="back" onNav={onNav} />
      <section className="empty" role="alert">
        <h2>정보를 불러오지 못했습니다</h2>
        <p>인터넷 연결을 확인한 뒤 다시 시도해 주세요.</p>
        <button className="btn btn-outline" style={{ marginTop: 32 }} onClick={onRetry}>
          다시 불러오기
        </button>
      </section>
    </div>
  );
}
