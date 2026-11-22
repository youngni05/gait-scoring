/** 01 로그인 · 01-1 입력 오류 · 01-2 로그인 실패 */
import { useState, type FormEvent } from 'react';
import { AlertCircleIcon, EyeIcon, EyeOffIcon, WalkIcon } from '../components/icons';
import { LoginFailedError, apiSource } from '../data';
import { errorMessage } from '../../../../mobile/src/api/client';
import { useNavigate } from '../router';
import { useSession } from '../session';

export function LoginPage({ renewal = false }: { renewal?: boolean }) {
  const { login } = useSession();
  const navigate = useNavigate();
  const [uid, setUid] = useState('');
  const [pw, setPw] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [errors, setErrors] = useState<{ id?: boolean; pw?: boolean }>({});
  /** 실패 안내 문구. null이면 안내를 숨긴다. */
  const [failed, setFailed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [server, setServer] = useState(apiSource.base);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    const next = { id: !uid.trim(), pw: !pw };
    setErrors(next);
    if (next.id || next.pw) return;
    setBusy(true);
    setFailed(null);
    try { if (!renewal) apiSource.configure(server); }
    catch (e) { setFailed(errorMessage(e)); setBusy(false); return; }
    login(uid.trim(), pw)
      .then(() => { if (!renewal) navigate('/subjects', { replace: true }); })
      .catch((err) => {
        setFailed(
          err instanceof LoginFailedError
            ? '아이디 또는 비밀번호가 올바르지 않습니다. 다시 확인해 주세요.'
            : errorMessage(err),
        );
        setPw('');
      })
      .finally(() => setBusy(false));
  };

  return (
    <main className="screen" style={{ padding: '80px 24px 40px', display: 'flex', flexDirection: 'column' }}>
      <section style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
        <div
          style={{ width: 80, height: 80, borderRadius: 'var(--r-lg)', background: 'var(--primary)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <WalkIcon size={40} />
        </div>
        <h1 style={{ marginTop: 24, fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em' }}>자동보행분석기</h1>
        <p style={{ marginTop: 8, fontSize: 16 }} className="muted">보행 영상 분석 · 결과 조회</p>
      </section>

      {failed && (
        <div className="alert alert-danger" role="alert" style={{ marginTop: 32 }}>
          <AlertCircleIcon size={22} />
          <div>
            <p className="title">로그인에 실패했습니다</p>
            <p style={{ marginTop: 4 }}>{failed}</p>
          </div>
        </div>
      )}

      <form onSubmit={submit} noValidate style={{ marginTop: failed ? 24 : 40, display: 'grid', gap: 20 }}>
        {!renewal && <div className="field"><label htmlFor="api-server">서버 주소</label><input id="api-server" className="input" type="url" value={server} disabled={busy} onChange={e => setServer(e.target.value)} /></div>}
        <div className="field">
          <label htmlFor="uid" style={{ fontWeight: 600 }}>아이디</label>
          <input
            id="uid"
            className={`input ${errors.id ? 'invalid' : ''}`}
            autoComplete="username"
            placeholder="아이디를 입력하세요"
            value={uid}
            aria-invalid={errors.id || undefined}
            aria-describedby={errors.id ? 'err-id' : undefined}
            onChange={(e) => {
              setUid(e.target.value);
              if (errors.id) setErrors((p) => ({ ...p, id: false }));
            }}
          />
          {errors.id && (
            <p id="err-id" className="field-error"><AlertCircleIcon size={16} />아이디를 입력해 주세요.</p>
          )}
        </div>
        <div className="field">
          <label htmlFor="upw" style={{ fontWeight: 600 }}>비밀번호</label>
          <div className="input-wrap">
            <input
              id="upw"
              type={showPw ? 'text' : 'password'}
              className={`input pad-right ${errors.pw ? 'invalid' : ''}`}
              style={failed && !errors.pw ? { borderColor: 'var(--ring)' } : undefined}
              autoComplete="current-password"
              placeholder={failed ? '비밀번호를 다시 입력하세요' : '비밀번호를 입력하세요'}
              value={pw}
              aria-invalid={errors.pw || undefined}
              aria-describedby={errors.pw ? 'err-pw' : undefined}
              onChange={(e) => {
                setPw(e.target.value);
                if (errors.pw) setErrors((p) => ({ ...p, pw: false }));
              }}
            />
            <button
              type="button"
              className="icon-btn input-action"
              aria-label={showPw ? '비밀번호 숨기기' : '비밀번호 보기'}
              onClick={() => setShowPw((v) => !v)}
            >
              {showPw ? <EyeOffIcon /> : <EyeIcon />}
            </button>
          </div>
          {errors.pw && (
            <p id="err-pw" className="field-error"><AlertCircleIcon size={16} />비밀번호를 입력해 주세요.</p>
          )}
        </div>
        <button type="submit" className="btn btn-primary" style={{ marginTop: 12 }} disabled={busy}>
          {busy ? '확인 중…' : '로그인'}
        </button>
      </form>

      <p style={{ marginTop: 'auto', paddingTop: 40, textAlign: 'center', fontSize: 15, lineHeight: 1.6 }} className="muted">
        계정 발급 및 비밀번호 문의는
        <br />
        소속 기관 관리자에게 연락해 주세요.
      </p>
    </main>
  );
}
