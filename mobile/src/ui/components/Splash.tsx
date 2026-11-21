/**
 * 앱을 처음 열 때 잠깐 보이는 시작 화면.
 *
 * 흩어진 로고 조각이 돌며 모이고 앱 이름이 이어서 나타난 뒤, 1.6초가 지나면 흐려지며 사라진다.
 * 화면을 누르면 바로 넘어간다. 시작 화면이 떠 있는 동안에도 뒤에서는 로그인·대상자 목록 화면이
 * 평소처럼 준비되므로, 이 화면 때문에 데이터 불러오기가 늦어지지는 않는다.
 * '움직임 줄이기' 설정에서는 움직이지 않는 로고를 0.6초만 보여준다.
 */
import { useEffect, useState } from 'react';
import { APP_NAME, APP_TAGLINE, IntroMark } from './brand';

const SHOW_MS = 1600;
const SHOW_MS_REDUCED = 600;
const FADE_MS = 250;

export function Splash() {
  const [phase, setPhase] = useState<'show' | 'leaving' | 'gone'>('show');

  useEffect(() => {
    if (phase === 'show') {
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const t = window.setTimeout(() => setPhase('leaving'), reduced ? SHOW_MS_REDUCED : SHOW_MS);
      return () => window.clearTimeout(t);
    }
    if (phase === 'leaving') {
      const t = window.setTimeout(() => setPhase('gone'), FADE_MS);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [phase]);

  if (phase === 'gone') return null;
  return (
    <div
      className={`splash ${phase === 'leaving' ? 'leaving' : ''}`}
      role="presentation"
      onClick={() => setPhase('leaving')}
    >
      <IntroMark size={168} />
      <div className="splash-name">
        <h1>{APP_NAME}</h1>
        <p>{APP_TAGLINE}</p>
      </div>
      <div className="splash-company" aria-hidden="true">
        <b>해피에이징</b>
        <span>HAPPY AGING</span>
      </div>
    </div>
  );
}
