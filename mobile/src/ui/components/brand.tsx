/**
 * 해피에이징 로고와, 로고를 움직여 만든 로딩 아이콘.
 *
 * 로고는 네 색 고리 조각(초록·하늘·분홍·주황)으로 이루어지고,
 * 원본처럼 정사각형 배치에서 왼쪽(반시계 방향)으로 약 18° 기울어져 있다.
 * 로딩 아이콘은 이 조각들이 한 바퀴 돌며 벌어졌다가 다시 붙는다.
 * 한 바퀴의 시작과 끝이 원래 로고 모양이라, 움직임이 멈추면 로고로 보인다.
 * 움직임 정의(키프레임)는 ui.css의 「로고·로딩 아이콘」 묶음에 있다.
 *
 * 로고 색은 로고와 로딩 표시에만 쓴다. 조각이 겹친 곳은 곱하기 합성(multiply)으로
 * 진하게 보이므로 흰색·연한 바탕 위에 둔다(진한 바탕에서는 검게 보인다).
 * 16px 이하에서는 네 조각이 구분되지 않으므로 로딩 아이콘은 20px 이상으로 쓴다.
 */

/** 앱 이름과 소개 문구. 이름을 바꿀 때는 이 값과 index.html·manifest.webmanifest를 함께 고친다. */
export const APP_NAME = '해피워크';
export const APP_TAGLINE = '보행 영상 분석 · 결과 조회';

/** 조각마다 [움직임 이름, 색, 고리 모양]. 좌표는 가운데가 (100, 100)인 200×200 기준이다. */
const PETALS = [
  ['ul', '#5DBB4F', 'M52.30 122.82A42 42 0 0 1 90.42 47.99'],
  ['ur', '#16C1E4', 'M77.18 52.30A42 42 0 0 1 152.01 90.42'],
  ['lr', '#EF4F9A', 'M147.70 77.18A42 42 0 0 1 109.58 152.01'],
  ['ll', '#F7784A', 'M122.82 147.70A42 42 0 0 1 47.99 109.58'],
] as const;

/** 움직이지 않는 로고. label을 주면 화면 낭독기가 로고 이름을 읽는다. */
export function BrandMark({ size = 40, label }: { size?: number; label?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="12 12 176 176"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {PETALS.map(([key, color, d]) => (
        <path key={key} className="hm-petal" stroke={color} d={d} />
      ))}
    </svg>
  );
}

/**
 * 로딩 아이콘. 문구와 함께 쓰므로 화면 낭독기에는 숨긴다(문구가 상태를 알린다).
 * 벌어진 조각이 잘리지 않도록 로고보다 여백을 넓게 잡는다.
 */
export function LoadingMark({ size = 22, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="-10 -10 220 220"
      aria-hidden="true"
      className={`hm-loading ${className ?? ''}`}
      style={{ flexShrink: 0 }}
    >
      {PETALS.map(([key, color, d]) => (
        <g key={key} className={`hm-${key}`}>
          <path className="hm-petal" stroke={color} d={d} />
        </g>
      ))}
    </svg>
  );
}

/** 시작 화면용 로고. 흩어진 조각이 돌며 모여 로고가 된다(한 번만 재생). */
export function IntroMark({ size = 168 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-20 -20 240 240" role="img" aria-label="해피에이징" className="hm-intro">
      {PETALS.map(([key, color, d]) => (
        <g key={key} className={`hm-in-${key}`}>
          <path className="hm-petal" stroke={color} d={d} />
        </g>
      ))}
    </svg>
  );
}
