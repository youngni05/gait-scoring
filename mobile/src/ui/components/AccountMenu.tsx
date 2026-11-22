import { useEffect, useRef, useState } from 'react';
import { UserIcon } from './icons';
import { useSession } from '../session';

export function AccountMenu() {
  const { username, logout } = useSession();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  return <div className="account-menu" ref={root}
    onBlur={e => { if (!e.currentTarget.contains(e.relatedTarget)) setOpen(false); }}
    onKeyDown={e => { if (e.key === 'Escape' && open) { e.preventDefault(); setOpen(false); trigger.current?.focus(); } }}>
    <button ref={trigger} className="account-trigger" type="button" aria-label="계정 메뉴"
      aria-expanded={open} aria-controls="account-panel" onClick={() => setOpen(value => !value)}>
      <UserIcon size={20} />
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true" className={open ? 'account-chevron open' : 'account-chevron'}>
        <path d="m3 4.5 3 3 3-3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
    {open && <div id="account-panel" className="account-panel" role="region" aria-label="로그인 계정">
      <div className="account-identity"><p>로그인 계정</p><strong>{username || '로그인됨'}</strong></div>
      <button type="button" className="account-logout" onClick={logout}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M9 4H5v16h4M10 12h11m-4-4 4 4-4 4" />
        </svg>
        로그아웃
      </button>
    </div>}
  </div>;
}
