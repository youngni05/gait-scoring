/** 02 대상자 목록 · 02-1 빈 상태 · 02-2 로딩 · 검색 결과 없음 */
import { useState } from 'react';
import { Avatar } from '../components/layout';
import { LoadingMark } from '../components/brand';
import { ChevronRightIcon, ClearIcon, ClockIcon, PlusIcon, RotateIcon, SearchIcon, UserPlusIcon, UsersIcon } from '../components/icons';
import { ageOn, dataSource, fmtDate } from '../data';
import { useNavigate } from '../router';
import { useAsync } from '../session';

export function SubjectListPage() {
  const navigate = useNavigate();
  const [q, setQ] = useState('');
  const { data, loading, error, reload } = useAsync((signal) => dataSource.listSubjects(signal), []);

  const list = data ?? [];
  const empty = !loading && !error && list.length === 0;
  const keyword = q.trim();
  const shown = keyword ? list.filter((s) => s.name.includes(keyword)) : list;

  return (
    <div className={`screen ${empty ? '' : 'has-cta'}`}>
      <header className="topbar" style={{ padding: '24px 20px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: '-0.02em' }}>대상자 목록</h1>
          {loading ? (
            <span className="skeleton" style={{ width: 56, height: 20, marginBottom: 4 }} />
          ) : (
            !error && <span style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }} className="muted">총 {list.length}명</span>
          )}
        </div>
        <div className="input-wrap" style={{ marginTop: 16, opacity: loading || empty ? 0.5 : 1 }}>
          <span className="input-prefix"><SearchIcon size={20} /></span>
          <input
            type="search"
            className="input pad-left pad-right"
            style={{ borderRadius: 'var(--r-lg)' }}
            placeholder="이름으로 검색"
            aria-label="이름으로 검색"
            value={q}
            disabled={loading || empty}
            onChange={(e) => setQ(e.target.value)}
          />
          {q && (
            <button className="icon-btn input-action" style={{ width: 40, height: 40, top: 8 }} aria-label="검색어 지우기" onClick={() => setQ('')}>
              <ClearIcon />
            </button>
          )}
        </div>
      </header>

      {loading && (
        <>
          <p role="status" style={{ padding: '20px 20px 0', fontSize: 15, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }} className="primary">
            <LoadingMark size={22} /> 대상자 목록을 불러오는 중입니다…
          </p>
          <ul style={{ padding: '16px 20px 0', display: 'grid', gap: 12 }} aria-hidden="true">
            {[80, 96, 64, 80, 96].map((w, i) => (
              <li key={i} className="list-btn" style={{ pointerEvents: 'none' }}>
                <span className="skeleton" style={{ width: 48, height: 48 }} />
                <span style={{ flex: 1, display: 'grid', gap: 8 }}>
                  <span className="skeleton" style={{ width: w, height: 16 }} />
                  <span className="skeleton" style={{ width: 128, height: 14 }} />
                  <span className="skeleton" style={{ width: 96, height: 12 }} />
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {!loading && Boolean(error) && (
        <section className="empty" role="alert">
          <h2>대상자 목록을 불러오지 못했습니다</h2>
          <p>인터넷 연결을 확인한 뒤 다시 시도해 주세요.</p>
          <button className="btn btn-outline" style={{ marginTop: 32 }} onClick={reload}>
            <RotateIcon /> 다시 불러오기
          </button>
        </section>
      )}

      {empty && (
        <section className="empty">
          <div className="empty-icon" style={{ background: 'var(--accent)', color: 'var(--primary)' }}>
            <UsersIcon size={44} />
            <span
              style={{ position: 'absolute', right: -4, bottom: -4, width: 40, height: 40, borderRadius: 999, background: 'var(--primary)', border: '4px solid #fff', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <PlusIcon size={16} />
            </span>
          </div>
          <h2>등록된 대상자가 없습니다</h2>
          <p>대상자를 먼저 등록하면<br />보행 영상을 분석할 수 있습니다.</p>
          <button className="btn btn-primary" style={{ marginTop: 40 }} onClick={() => navigate('/subjects/new')}>
            <UserPlusIcon /> 대상자 등록
          </button>
        </section>
      )}

      {!loading && !error && shown.length > 0 && (
        <ul style={{ padding: '16px 20px 0', display: 'grid', gap: 12 }}>
          {shown.map((s, i) => (
            <li key={s.id}>
              <button className="list-btn" onClick={() => navigate(`/subjects/${s.id}`)}>
                <Avatar name={s.name} size="md" tone={i % 2 === 0 ? 'a' : 'b'} />
                <span style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ display: 'block', fontSize: 18, fontWeight: 700 }}>{s.name}</span>
                  <span style={{ display: 'block', marginTop: 2, fontSize: 15, color: 'var(--fg-70)' }}>
                    {[s.sex,s.birth && ageOn(s.birth)+'세',s.heightCm && s.heightCm+'cm'].filter(Boolean).join(' · ') || '등록 정보 미확인'}
                  </span>
                  {s.latestMeasuredAt ? (
                    <span className="primary" style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 4, fontSize: 14, fontWeight: 600 }}>
                      <ClockIcon size={14} /> 최근 분석 {fmtDate(s.latestMeasuredAt)}
                    </span>
                  ) : (
                    <span style={{ display: 'block', marginTop: 4, fontSize: 14, color: 'var(--fg-50)' }}>분석 기록 없음</span>
                  )}
                </span>
                <ChevronRightIcon size={18} className="chev" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {!loading && !error && !empty && shown.length === 0 && (
        <section style={{ padding: '64px 32px 0', textAlign: 'center' }}>
          <div style={{ width: 64, height: 64, margin: '0 auto', borderRadius: 999, background: 'var(--card)', color: 'var(--fg-40)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <SearchIcon size={26} />
          </div>
          <p style={{ marginTop: 20, fontSize: 18, fontWeight: 700 }}>‘{keyword}’ 검색 결과가 없습니다</p>
          <p style={{ marginTop: 8, fontSize: 15 }} className="muted">이름을 다시 확인하거나 새 대상자를 등록해 주세요.</p>
        </section>
      )}

      {!empty && (
        <div className="bottom-cta">
          <button className="btn btn-primary" disabled={loading} onClick={() => navigate('/subjects/new')}>
            <UserPlusIcon /> 대상자 등록
          </button>
        </div>
      )}
    </div>
  );
}
