/**
 * 아주 작은 주소(#/경로) 기반 화면 전환기.
 * 팀 저장소에 새 라이브러리(react-router)를 들이지 않으려고 직접 만들었다.
 * 주소의 # 뒤를 바꾸면 브라우저 기록이 남으므로, 안드로이드 뒤로 가기 버튼으로도 이전 화면에 돌아간다.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import { LeaveGuards } from './leave-guards';

interface RouterValue {
  registerLeaveGuard: (guard: () => string | null) => () => void;
  confirmLeave: () => boolean;
  path: string;
  navigate: (to: string, opts?: { replace?: boolean }) => void;
}

const Ctx = createContext<RouterValue | null>(null);

const readPath = () => {
  const raw = window.location.hash.replace(/^#/, '');
  return raw.startsWith('/') ? raw : '/';
};

export function RouterProvider({ children }: { children: ReactNode }) {
  const [path, setPath] = useState(readPath);
  const acceptedPath = useRef(path);
  const acceptedIndex = useRef<number>(window.history.state?.gaitNavIndex ?? 0);
  const restoring = useRef(false);
  const guards = useMemo(() => new LeaveGuards(), []);
  const registerLeaveGuard = useCallback((guard: () => string | null) => guards.register(guard), [guards]);
  const confirmLeave = useCallback(() => guards.confirm(message => window.confirm(message)), [guards]);

  useEffect(() => {
    window.history.replaceState({ ...window.history.state, gaitNavIndex: acceptedIndex.current }, '');
    const onChange = () => {
      const next = readPath();
      if (restoring.current) { restoring.current = false; return; }
      const nextIndex: unknown = window.history.state?.gaitNavIndex;
      if (next === acceptedPath.current) {
        if (typeof nextIndex === 'number') acceptedIndex.current = nextIndex;
        return;
      }
      if (!confirmLeave()) {
        // Traverse back to the accepted entry; do not overwrite the user's history.
        restoring.current = true;
        window.history.go(typeof nextIndex === 'number' ? acceptedIndex.current - nextIndex : -1);
        return;
      }
      acceptedIndex.current = typeof nextIndex === 'number' ? nextIndex : acceptedIndex.current + 1;
      window.history.replaceState({ ...window.history.state, gaitNavIndex: acceptedIndex.current }, '');
      acceptedPath.current = next;
      setPath(next);
    };
    const onUnload = (event: BeforeUnloadEvent) => {
      if (guards.message()) { event.preventDefault(); event.returnValue = ''; }
    };
    window.addEventListener('hashchange', onChange);
    window.addEventListener('beforeunload', onUnload);
    return () => {
      window.removeEventListener('hashchange', onChange);
      window.removeEventListener('beforeunload', onUnload);
    };
  }, [confirmLeave, guards]);

  useEffect(() => { window.scrollTo(0, 0); }, [path]);

  const navigate = useCallback((to: string, opts?: { replace?: boolean }) => {
    if (to === acceptedPath.current || !confirmLeave()) return;
    // pushState avoids a second confirmation from a programmatic hashchange.
    if (opts?.replace) window.history.replaceState({ ...window.history.state, gaitNavIndex: acceptedIndex.current }, '', `#${to}`);
    else {
      acceptedIndex.current += 1;
      window.history.pushState({ ...window.history.state, gaitNavIndex: acceptedIndex.current }, '', `#${to}`);
    }
    acceptedPath.current = to;
    setPath(to);
  }, [confirmLeave]);

  const value = useMemo(() => ({ path, navigate, registerLeaveGuard, confirmLeave }),
    [path, navigate, registerLeaveGuard, confirmLeave]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useRouter(): RouterValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('RouterProvider 안에서만 사용할 수 있습니다.');
  return v;
}

export const useNavigate = () => useRouter().navigate;

/** "/subjects/:subjectId" 같은 모양과 현재 주소를 맞춰 보고, 맞으면 이름 붙은 값을 돌려준다. */
export function matchPath(pattern: string, path: string): Record<string, string> | null {
  const p = pattern.split('/').filter(Boolean);
  const s = path.split('/').filter(Boolean);
  if (p.length !== s.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < p.length; i++) {
    const seg = p[i]!;
    const val = s[i]!;
    if (seg.startsWith(':')) params[seg.slice(1)] = decodeURIComponent(val);
    else if (seg !== val) return null;
  }
  return params;
}

/** 현재 화면의 주소 값(대상자 번호 등). App에서 화면을 고를 때 채운다. */
export const ParamsContext = createContext<Record<string, string>>({});
export const useParams = () => useContext(ParamsContext);

/** Keep the latest condition without replacing a registered guard on every render. */
export function useLeaveGuard(blocked: boolean, message: string) {
  const { registerLeaveGuard } = useRouter();
  const current = useRef({ blocked, message });
  current.current = { blocked, message };
  useEffect(() => registerLeaveGuard(() => current.current.blocked ? current.current.message : null), [registerLeaveGuard]);
}
