// 해피워크 앱 아이콘(D안: 흰 바탕 + 원래 색 해피에이징 로고 + 오른쪽 아래 하늘색 발자국 배지)을
// PNG·SVG 파일로 만든다. 외부 패키지 없이 Node 기본 기능(zlib)만 쓴다. 저장소 루트에서 실행한다.
//   node mobile/brand/render-icons.mjs
// 결과: mobile/brand/ 아래 원본 파일들과, PWA가 쓰는 mobile/public/ 아이콘 파일들.
import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const brandDir = here;
const publicDir = join(here, '..', 'public');

const WHITE = [255, 255, 255];
const BLUE = [0x1e, 0x9d, 0xf1]; // 앱 primary(ui.css --primary)

// ── 모양 정의(좌표는 로고 가운데가 (100,100)인 단위) ──
// 로고: 반지름 42, 두께 26인 고리 조각 네 개. 원본처럼 정사각형 배치에서 왼쪽으로 18° 기울어져 있고,
// 조각 끝이 이웃 조각 위에 겹친다. 겹친 곳은 곱하기 합성으로 진하게 보인다(components/brand.tsx와 같은 모양).
const R = 42;
const HALF = 13;
const ARCS = [
  { cx: 72.28, cy: 85.88, from: 118.4, color: [0x5d, 0xbb, 0x4f] },
  { cx: 114.12, cy: 72.28, from: 208.4, color: [0x16, 0xc1, 0xe4] },
  { cx: 127.72, cy: 114.12, from: 298.4, color: [0xef, 0x4f, 0x9a] },
  { cx: 85.88, cy: 127.72, from: 28.4, color: [0xf7, 0x78, 0x4a] },
];
const SPAN = 177.2;
// 발자국 배지: 가운데 (170,170), 하늘색 원(반지름 32) + 흰 테두리(32~40) + 흰 신발 자국 두 개.
// 신발 자국 하나 = 앞창(넓은 앞쪽 타원 + 좁아지는 아치 타원) + 조금 떨어진 반원 뒤축.
// 두 자국을 엇갈려 놓고 -25° 기울여 왼쪽 위(로고 쪽)로 걸어가게 한다(components/brand.tsx FootBadge와 같은 값).
const BADGE = { cx: 170, cy: 170, inner: 32, outer: 40 };
const SHOE = {
  // 뒷발은 오른쪽 아래, 앞발은 왼쪽 위(걷는 방향 기준 앞뒤 8, 좌우 8씩 벌림)
  at: [[180.6, 174.9], [159.4, 167.1]],
  rot: -25,
  scale: 0.8,
  ellipses: [[0, -8, 8.5, 13], [0.5, 2, 6.5, 10]],
  heel: { cx: 0, cy: 15.5, r: 6.5, top: 14.5 }, // 원의 top보다 아래쪽만 남겨 반원 뒤축을 만든다
};
// 로고와 배지를 모두 담는 범위: -14 ~ 214 (폭 228 단위)
const CONTENT_SPAN = 228;

function arcColors(u, v) {
  const hits = [];
  for (const a of ARCS) {
    const dx = u - a.cx;
    const dy = v - a.cy;
    if (Math.abs(Math.hypot(dx, dy) - R) > HALF) continue;
    const ang = (Math.atan2(dy, dx) * 180) / Math.PI;
    const rel = (((ang - a.from) % 360) + 360) % 360;
    if (rel <= SPAN) hits.push(a.color);
  }
  return hits;
}
function inSole(u, v) {
  const t = (SHOE.rot * Math.PI) / 180;
  for (const [ax, ay] of SHOE.at) {
    const dx = (u - ax) / SHOE.scale;
    const dy = (v - ay) / SHOE.scale;
    const x = dx * Math.cos(t) + dy * Math.sin(t);
    const y = -dx * Math.sin(t) + dy * Math.cos(t);
    for (const [cx, cy, rx, ry] of SHOE.ellipses) if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1) return true;
    const h = SHOE.heel;
    if (y >= h.top && (x - h.cx) ** 2 + (y - h.cy) ** 2 <= h.r * h.r) return true;
  }
  return false;
}

/** 표본 한 점의 색. bg가 있으면 바탕색, 없으면 null(투명)에서 시작한다. */
function sampleColor(u, v, bg) {
  let c = bg ? [...bg] : null;
  for (const col of arcColors(u, v)) {
    c = c ? c.map((n, i) => Math.round((n * col[i]) / 255)) : [...col];
  }
  const d = Math.hypot(u - BADGE.cx, v - BADGE.cy);
  if (d <= BADGE.outer) c = d <= BADGE.inner && !inSole(u, v) ? [...BLUE] : [...WHITE];
  return c;
}

/**
 * size: 출력 픽셀, ratio: 내용(로고+배지) 폭 / 전체 폭,
 * shape: 'round'(흰 둥근 사각, 바깥 투명) | 'full'(흰 꽉 찬 사각) | 'none'(바탕 없음)
 */
function render({ size, ratio, shape, radiusRatio = 0.225 }) {
  const SS = 4; // 픽셀당 4×4 표본으로 가장자리를 부드럽게 한다
  const k = (ratio * size) / CONTENT_SPAN;
  const rad = radiusRatio * size;
  const px = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const fx = x + (sx + 0.5) / SS;
          const fy = y + (sy + 0.5) / SS;
          let bg = shape !== 'none';
          if (shape === 'round') {
            const qx = Math.max(rad - fx, 0, fx - (size - rad));
            const qy = Math.max(rad - fy, 0, fy - (size - rad));
            bg = qx * qx + qy * qy <= rad * rad;
          }
          const u = 100 + (fx - size / 2) / k;
          const v = 100 + (fy - size / 2) / k;
          const c = sampleColor(u, v, bg ? WHITE : null);
          if (c) { r += c[0]; g += c[1]; b += c[2]; a += 1; }
        }
      }
      const i = (y * size + x) * 4;
      if (a) {
        px[i] = Math.round(r / a); px[i + 1] = Math.round(g / a); px[i + 2] = Math.round(b / a);
        px[i + 3] = Math.round((255 * a) / (SS * SS));
      }
    }
  }
  return encodePng(size, size, px);
}

// ── PNG 저장 ──
const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function encodePng(w, h, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) rgba.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0)),
  ]);
}

// ── SVG 원본 ──
const PATHS = [
  ['#5DBB4F', 'M52.30 122.82A42 42 0 0 1 90.42 47.99'],
  ['#16C1E4', 'M77.18 52.30A42 42 0 0 1 152.01 90.42'],
  ['#EF4F9A', 'M147.70 77.18A42 42 0 0 1 109.58 152.01'],
  ['#F7784A', 'M122.82 147.70A42 42 0 0 1 47.99 109.58'],
];
const HEEL_HALF = Math.sqrt(SHOE.heel.r ** 2 - (SHOE.heel.cy - SHOE.heel.top) ** 2).toFixed(2);
const SHOE_SVG = SHOE.ellipses.map(([cx, cy, rx, ry]) => `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"/>`).join('')
  + `<path d="M${HEEL_HALF} ${SHOE.heel.top}A${SHOE.heel.r} ${SHOE.heel.r} 0 1 1 -${HEEL_HALF} ${SHOE.heel.top}Z"/>`;
const FOOT_SVG = SHOE.at
  .map(([x, y]) => `<g fill="#FFFFFF" transform="translate(${x} ${y}) rotate(${SHOE.rot}) scale(${SHOE.scale})">${SHOE_SVG}</g>`)
  .join('');
function svg({ ratio, shape, radiusRatio = 0.225 }) {
  const size = 1024;
  const k = (ratio * size) / CONTENT_SPAN;
  const t = `translate(${size / 2} ${size / 2}) scale(${k.toFixed(5)}) translate(-100 -100)`;
  const bg = shape === 'round'
    ? `<rect width="${size}" height="${size}" rx="${radiusRatio * size}" fill="#FFFFFF"/>`
    : shape === 'full' ? `<rect width="${size}" height="${size}" fill="#FFFFFF"/>` : '';
  const logo = PATHS.map(([c, d]) => `<path d="${d}" stroke="${c}" style="mix-blend-mode:multiply"/>`).join('');
  const badge = `<circle cx="170" cy="170" r="36" fill="#1E9DF1" stroke="#FFFFFF" stroke-width="8"/>` + FOOT_SVG;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${bg}<g transform="${t}"><g fill="none" stroke-width="26" stroke-linecap="butt">${logo}</g>${badge}</g></svg>\n`;
}

const out = (dir, name, data) => { mkdirSync(dir, { recursive: true }); writeFileSync(join(dir, name), data); console.log('wrote', join(dir, name)); };

// 내용 크기: 일반 아이콘은 폭의 84%. 마스크용·안드로이드 전경은 기기가 잘라도 배지까지 남도록 안전 영역 안에 둔다.
const ANY = { ratio: 0.84, shape: 'round' };
const MASK = { ratio: 0.62, shape: 'full' };
const APPLE = { ratio: 0.74, shape: 'full' };
const FG = { ratio: 0.5, shape: 'none' };

// 원본(나중에 앱 개발에서 다시 쓰는 파일)
out(brandDir, 'happywalk-icon.svg', svg(ANY));
out(brandDir, 'happywalk-icon-maskable.svg', svg(MASK));
out(brandDir, 'happywalk-icon-foreground.svg', svg(FG));
out(brandDir, 'happywalk-icon-1024.png', render({ size: 1024, ...ANY }));
out(brandDir, 'happywalk-icon-maskable-1024.png', render({ size: 1024, ...MASK }));
out(brandDir, 'happywalk-icon-foreground-432.png', render({ size: 432, ...FG }));

// PWA가 쓰는 파일
out(publicDir, 'icon-192.png', render({ size: 192, ...ANY }));
out(publicDir, 'icon-512.png', render({ size: 512, ...ANY }));
out(publicDir, 'icon-maskable-512.png', render({ size: 512, ...MASK }));
out(publicDir, 'apple-touch-icon.png', render({ size: 180, ...APPLE }));
out(publicDir, 'favicon.svg', svg(ANY));
