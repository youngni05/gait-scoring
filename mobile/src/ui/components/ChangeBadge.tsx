import type { Verdict } from '../data/visit-summary.mjs';

export function ChangeBadge({ delta, verdict }: { delta: number; verdict: Verdict }) {
  if (verdict === 'similar') return <span className="change-badge similar">비슷함</span>;
  const up = delta > 0;
  return <span className={'change-badge ' + (up ? 'up' : 'down')}>
    <span aria-hidden="true">{up ? '↑' : '↓'}</span> {up ? '증가' : '감소'}
  </span>;
}
