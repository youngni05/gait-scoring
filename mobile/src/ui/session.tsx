/** 로그인 상태와 비동기 데이터 불러오기 도우미. */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { dataSource } from './data';
import { AUTH_EXPIRED } from './data/apiSource';

interface Session {
  loggedIn: boolean;
  expired: boolean;
  username: string;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const Ctx = createContext<Session | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [loggedIn, setLoggedIn] = useState(false);
  const [expired, setExpired] = useState(false);
  const [username, setUsername] = useState('');
  useEffect(() => {
    const expire = () => setExpired(true);
    window.addEventListener(AUTH_EXPIRED, expire);
    return () => window.removeEventListener(AUTH_EXPIRED, expire);
  }, []);
  const login = useCallback(async (u: string, p: string) => {
    await dataSource.login(u, p);
    setUsername(u);
    setLoggedIn(true);
    setExpired(false);
  }, []);
  const logout = useCallback(() => {
    dataSource.logout();
    setUsername('');
    setLoggedIn(false);
    setExpired(false);
  }, []);
  const value = useMemo(() => ({ loggedIn, expired, username, login, logout }), [loggedIn, expired, username, login, logout]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession(): Session {
  const v = useContext(Ctx);
  if (!v) throw new Error('SessionProvider 안에서만 사용할 수 있습니다.');
  return v;
}

export interface AsyncState<T> {
  data: T | undefined;
  loading: boolean;
  error: unknown;
  reload: () => void;
}

/**
 * 화면이 열릴 때 데이터를 불러온다. 화면을 떠나거나 조건이 바뀌면 이전 요청을 취소한다.
 * deps가 바뀌면 다시 불러온다.
 */
export function useAsync<T>(load: (signal: AbortSignal) => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const abort = new AbortController();
    setLoading(true);
    setError(null);
    load(abort.signal)
      .then((v) => {
        if (!abort.signal.aborted) setData(v);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(e);
      })
      .finally(() => {
        if (!abort.signal.aborted) setLoading(false);
      });
    return () => abort.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { data, loading, error, reload: () => setTick((n) => n + 1) };
}
