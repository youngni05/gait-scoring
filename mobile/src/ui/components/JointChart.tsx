/**
 * 관절 각도 그래프(왼쪽 실선 · 오른쪽 점선).
 * 디자인 시안은 Plotly를 썼지만, 팀 저장소에 새 라이브러리를 들이지 않으려고 SVG로 직접 그린다.
 * - 그래프를 누르거나 손가락을 올리면 그 지점의 좌우 각도를 보여준다.
 * - zoomable이면 가로로 드래그한 구간만 확대한다(07-2 그래프 확대 보기).
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { CYCLE_X, type JointCurve } from '../data';

const C = { L: '#1e9df1', R: '#e0245e', grid: '#e1eaef', zero: '#cfd3d6', fg: '#0f1419', sub: 'rgb(15 20 25 / .6)' };
const M = { l: 48, r: 14, t: 10, b: 44 };

interface Props {
  curve: JointCurve | {left:(number|null)[];right:(number|null)[]};
  native?: {x:number[];source_indices:number[]};
  height: number;
  zoomable?: boolean;
  /** 값이 바뀌면 확대를 풀고 처음 범위로 돌아간다. */
  resetKey?: number;
  label: string;
}

/** 눈금 간격을 1·2·5 단위의 보기 좋은 수로 고른다. */
function niceStep(span: number, target: number): number {
  const raw = span / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  const n = raw / p;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
}

/** 점들을 부드러운 곡선(Catmull-Rom → 3차 베지어)으로 잇는다. */
function smoothPath(pts: [number, number][]): string {
  if (pts.length < 2) return '';
  let d = `M${pts[0]![0]},${pts[0]![1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i]!;
    const p1 = pts[i]!;
    const p2 = pts[i + 1]!;
    const p3 = pts[i + 2] ?? p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x},${c1y} ${c2x},${c2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

let clipSeq = 0;

export function JointChart({ curve, native, height, zoomable = false, resetKey = 0, label }: Props) {
  const axis = native?.x ?? CYCLE_X;
  const end = axis.at(-1) || 1;
  const boxRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(300);
  const [domain, setDomain] = useState<[number, number]>([0, end]);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useMemo(() => `jc-clip-${++clipSeq}`, []);

  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, Math.round(e!.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setDomain([0, end]);
    setHover(null);
  }, [curve, end, resetKey]);

  const big = zoomable;
  const plotW = width - M.l - M.r;
  const plotH = height - M.t - M.b;
  const [x0, x1] = domain;

  // 보이는 구간의 값으로 세로축 범위를 정한다.
  const visible = axis.map((x, i) => ({ x, l: curve.left[i], r: curve.right[i] })).filter((p) => p.x >= x0 && p.x <= x1);
  const ys = visible.flatMap((p) => [p.l, p.r]).filter((v): v is number => typeof v === 'number');
  const rawMin = Math.min(...ys, 0);
  const rawMax = Math.max(...ys, 0);
  const yStep = niceStep(rawMax - rawMin || 10, big ? 8 : 5);
  const y0 = Math.floor(rawMin / yStep) * yStep;
  const y1 = Math.max(y0 + yStep, Math.ceil(rawMax / yStep) * yStep);
  const xStep = x1 - x0 === 100 ? (big ? 10 : 25) : niceStep(x1 - x0, big ? 6 : 4);

  const sx = (x: number) => M.l + ((x - x0) / (x1 - x0)) * plotW;
  const sy = (y: number) => M.t + (1 - (y - y0) / (y1 - y0)) * plotH;
  const toX = (px: number) => x0 + ((px - M.l) / plotW) * (x1 - x0);

  const xTicks: number[] = [];
  for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1 + 1e-9; v += xStep) xTicks.push(Math.round(v * 100) / 100);
  const yTicks: number[] = [];
  for (let v = y0; v <= y1 + 1e-9; v += yStep) yTicks.push(Math.round(v * 100) / 100);

  const line = (vals: (number|null)[]) => {
    if (!native) return smoothPath(axis.map((x,i) => [sx(x),sy(vals[i]!)]));
    return axis.map((x,i) => {
      const v = vals[i];
      if (v == null) return '';
      const connected = vals[i-1] != null && native.source_indices[i] === native.source_indices[i-1]! + 1;
      return (connected ? 'L' : 'M') + sx(x) + ',' + sy(v);
    }).join(' ');
  };

  // ── 손가락·마우스 처리 ──
  const localX = (e: PointerEvent) => {
    const rect = boxRef.current!.getBoundingClientRect();
    return Math.min(Math.max(e.clientX - rect.left, M.l), M.l + plotW);
  };
  const nearestIndex = (px: number) => {
    const x = toX(px);
    return axis.reduce((best,v,i)=>Math.abs(v-x)<Math.abs(axis[best]!-x)?i:best,0);
  };

  const onDown = (e: PointerEvent) => {
    if (!zoomable) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const px = localX(e);
    setDrag({ a: px, b: px });
  };
  const onMove = (e: PointerEvent) => {
    const px = localX(e);
    if (drag) setDrag({ ...drag, b: px });
    else setHover(nearestIndex(px));
  };
  const onUp = (e: PointerEvent) => {
    if (!drag) {
      setHover(nearestIndex(localX(e)));
      return;
    }
    const a = Math.min(drag.a, drag.b);
    const b = Math.max(drag.a, drag.b);
    setDrag(null);
    if (b - a < 12) {
      setHover(nearestIndex(localX(e)));
      return; // 짧은 탭은 확대가 아니라 값 보기로 처리한다.
    }
    const lo = Math.max(0, native ? toX(a) : Math.floor(toX(a)));
    const hi = Math.min(end, native ? toX(b) : Math.ceil(toX(b)));
    if (hi - lo >= end / 25) setDomain([lo, hi]);
  };

  const hv = hover !== null ? { x: axis[hover]!, l: curve.left[hover], r: curve.right[hover] } : null;
  const showHover = hv && hv.x >= x0 && hv.x <= x1;
  const fs = big ? 13 : 12;

  return (
    <div
      ref={boxRef}
      style={{ position: 'relative', height, touchAction: zoomable ? 'none' : 'pan-y', userSelect: 'none' }}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerLeave={() => !drag && setHover(null)}
      onPointerCancel={() => setDrag(null)}
    >
      <svg width={width} height={height} role="img" aria-label={`${label} 관절 각도 그래프. 왼쪽 실선, 오른쪽 점선.`}>
        <defs>
          <clipPath id={clipId}>
            <rect x={M.l} y={M.t} width={plotW} height={plotH} />
          </clipPath>
        </defs>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={M.l} x2={M.l + plotW} y1={sy(v)} y2={sy(v)} stroke={v === 0 ? C.zero : C.grid} strokeWidth={v === 0 ? 2 : 1} />
            <text x={M.l - 8} y={sy(v)} dy="0.35em" textAnchor="end" fontSize={fs} fill={C.fg}>{v}</text>
          </g>
        ))}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line x1={sx(v)} x2={sx(v)} y1={M.t} y2={M.t + plotH} stroke={C.grid} />
            <text x={sx(v)} y={M.t + plotH + 18} textAnchor="middle" fontSize={fs} fill={C.fg}>{v}</text>
          </g>
        ))}
        <text x={M.l + plotW / 2} y={height - 6} textAnchor="middle" fontSize={12} fill={C.fg}>{native ? '구간 경과 시간 (초)' : '보행 주기 (%)'}</text>
        <text transform={`translate(13 ${M.t + plotH / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill={C.fg}>각도 (°)</text>

        <g clipPath={`url(#${clipId})`}>
          <path d={line(curve.left)} fill="none" stroke={C.L} strokeWidth={3} strokeLinecap="round" />
          <path d={line(curve.right)} fill="none" stroke={C.R} strokeWidth={3} strokeDasharray="7 4" />
          {showHover && (
            <>
              <line x1={sx(hv.x)} x2={sx(hv.x)} y1={M.t} y2={M.t + plotH} stroke={C.sub} strokeDasharray="3 3" />
              {hv.l != null && <circle cx={sx(hv.x)} cy={sy(hv.l)} r={4.5} fill="#fff" stroke={C.L} strokeWidth={2} />}
              {hv.r != null && <circle cx={sx(hv.x)} cy={sy(hv.r)} r={4.5} fill="#fff" stroke={C.R} strokeWidth={2} />}
            </>
          )}
        </g>
        {drag && (
          <rect
            x={Math.min(drag.a, drag.b)}
            y={M.t}
            width={Math.abs(drag.b - drag.a)}
            height={plotH}
            fill="rgb(30 157 241 / .15)"
            stroke={C.L}
          />
        )}
      </svg>

      {showHover && !drag && (
        <div
          role="status"
          style={{
            position: 'absolute',
            top: M.t + 4,
            left: Math.min(Math.max(sx(hv.x) - 64, 4), width - 132),
            width: 128,
            pointerEvents: 'none',
            background: 'rgb(15 20 25 / .85)',
            color: '#fff',
            borderRadius: 8,
            padding: '6px 10px',
            fontSize: 'var(--text-caption)',
            lineHeight: 1.5,
          }}
        >
          <b>{native ? hv.x.toFixed(3) + '초' : '보행 주기 ' + hv.x + '%'}</b>
          <br />
          <span style={{ color: '#8fd0ff' }}>■</span> 왼쪽 {hv.l == null ? '관측 불가' : hv.l.toFixed(1)+'°'}
          <br />
          <span style={{ color: '#ff8fb0' }}>■</span> 오른쪽 {hv.r == null ? '관측 불가' : hv.r.toFixed(1)+'°'}
        </div>
      )}
    </div>
  );
}
