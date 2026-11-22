/**
 * 아주 작은 주소(#/경로) 기반 화면 전환기.
 * 팀 저장소에 새 라이브러리(react-router)를 들이지 않으려고 직접 만들었다.
 * 주소의 # 뒤를 바꾸면 브라우저 기록이 남으므로, 안드로이드 뒤로 가기 버튼으로도 이전 화면에 돌아간다.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

interface RouterValue {
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

  useEffect(() => {
    const onChange = () => setPath(readPath());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  // 화면이 바뀌면 맨 위부터 보여준다.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [path]);

  const navigate = useCallback((to: string, opts?: { replace?: boolean }) => {
    if (opts?.replace) {
      window.history.replaceState(window.history.state, '', `#${to}`);
      setPath(to);
    } else {
      window.location.hash = to;
    }
  }, []);

  const value = useMemo(() => ({ path, navigate }), [path, navigate]);
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
