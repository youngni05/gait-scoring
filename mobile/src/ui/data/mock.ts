/**
 * 화면 초안용 예시 데이터와 날짜·나이 계산.
 * 실제 서버·분석 모델과 연결되지 않았다. 숫자와 그래프는 모두 가상 값이다.
 */

export type Sex = '여' | '남';
export type Joint = 'hip' | 'knee' | 'ankle';
export const JOINTS: Joint[] = ['hip', 'knee', 'ankle'];
export const JOINT_LABEL: Record<Joint, string> = { hip: '고관절', knee: '무릎', ankle: '발목' };

export interface Subject {
  id: string;
  name: string;
  sex: Sex | null;
  /** YYYY-MM-DD */
  birth: string;
  heightCm: number | null;
}

/** 보행 주기 0~100%를 2% 간격(51점)으로 나눈 좌우 관절 각도(°). */
export interface JointCurve {
  left: number[];
  right: number[];
}

/** 산출하지 못한 항목과 그 이유. 값이 있으면 그 항목은 결과지에 ‘—’로 표시한다. */
export type MissingKey = 'symmetry' | Joint;

export interface ResultRecord {
  id: string;
  subjectId: string;
  /** ISO 날짜·시각 */
  measuredAt: string;
  captureDay?: string | null;
  dateBasis?: 'analysis_start' | 'legacy_capture' | 'capture';
  /** 촬영 당시 나이와 키. 대상자 정보를 나중에 고쳐도 이 값은 바뀌지 않는다. */
  ageAtCapture: number | null;
  /** Versioned JSON returned by the existing Python report adapter. */
  live?: any;
  serverState?: 'pending' | 'failed' | 'ready' | 'legacy';
  heightAtCapture: number | null;
  gdi: number | null;
  cadence: number | null;
  symmetry: number | null;
  curves: Partial<Record<Joint, JointCurve>>;
  missing: Partial<Record<MissingKey, string>>;
  fileName: string;
}

// ── 날짜·나이 ────────────────────────────────

const WEEKDAY = ['일', '월', '화', '수', '목', '금', '토'];

/** 만 나이. 생일이 지나지 않았으면 한 살을 뺀다. */
export function ageOn(birth: string, on: Date = new Date()): number {
  const [y, m, d] = birth.split('-').map(Number) as [number, number, number];
  let age = on.getFullYear() - y;
  const beforeBirthday = on.getMonth() + 1 < m || (on.getMonth() + 1 === m && on.getDate() < d);
  if (beforeBirthday) age -= 1;
  return age;
}

const pad = (n: number) => String(n).padStart(2, '0');

// 촬영 날짜는 한국 날짜로 표시한다. 시간대가 없는 예시 기록은 입력된 날짜를 유지한다.
const seoulDate = new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Seoul' });
const seoulTime = new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', hour: 'numeric', minute: '2-digit', hour12: true });
const hasOffset = (iso: string) => /(?:Z|[+-]\d{2}:\d{2})$/i.test(iso);
export const displayDay = (iso: string) => hasOffset(iso) ? seoulDate.format(new Date(iso)) : iso.slice(0, 10);
export const fmtDate = (iso: string) => iso ? displayDay(iso).replaceAll('-', '.') : '촬영일 미확인';
export const fmtWeekday = (iso: string) => iso ? WEEKDAY[new Date(displayDay(iso) + 'T12:00:00').getDay()] : '';
export const fmtTime = (iso: string) => {
  if (!iso) return "";
  if (hasOffset(iso)) return seoulTime.format(new Date(iso));
  const d = new Date(iso);
  const h = d.getHours();
  return `${h < 12 ? '오전' : '오후'} ${h % 12 === 0 ? 12 : h % 12}:${pad(d.getMinutes())}`;
};
export const fmtBirth = (birth: string) => birth.replaceAll('-', '.');
export const fmtCompact = (iso: string) => fmtDate(iso).replaceAll('.', '');

/** "1947.05.14" 같은 입력을 검사해 "1947-05-14"로 돌려준다. 틀리면 null. */
export function parseBirth(text: string, today: Date = new Date()): string | null {
  const m = /^(\d{4})\.(\d{2})\.(\d{2})$/.exec(text.trim());
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(y, mo - 1, d);
  if (date.getFullYear() !== y || date.getMonth() !== mo - 1 || date.getDate() !== d) return null;
  if (date > today || y < 1900) return null;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

/** 숫자만 받아 YYYY.MM.DD 모양으로 점을 넣는다. */
export function maskBirth(raw: string): string {
  const digits = raw.replace(/\D/g, '').slice(0, 8);
  if (digits.length <= 4) return digits;
  if (digits.length <= 6) return `${digits.slice(0, 4)}.${digits.slice(4)}`;
  return `${digits.slice(0, 4)}.${digits.slice(4, 6)}.${digits.slice(6)}`;
}

// ── 관절 곡선(가상) ───────────────────────────
// 디자인 시안의 Plotly 예시 식을 그대로 옮겼다. 정상 보행 곡선의 대략적인 모양만 흉내 낸다.

const g = (t: number, m: number, s: number) => Math.exp(-0.5 * ((t - m) / s) ** 2);
const SHAPE: Record<Joint, (t: number, shift: number, k: number) => number> = {
  hip: (t, s, k) => 12 + 21 * k * Math.cos(2 * Math.PI * (t - 0.9 - s)),
  knee: (t, s, k) => 5 + k * (13 * g(t, 0.15 + s, 0.07) + 55 * g(t, 0.72 + s, 0.11)),
  ankle: (t, s, k) => 2 + k * (-6 * g(t, 0.07 + s, 0.04) + 10 * g(t, 0.45 + s, 0.12) - 20 * g(t, 0.64 + s, 0.06)),
};
export const CYCLE_X = Array.from({ length: 51 }, (_, i) => i * 2);

/** asym이 클수록 오른쪽 곡선이 왼쪽과 더 어긋난다. */
export function makeCurve(joint: Joint, asym: number): JointCurve {
  const f = SHAPE[joint];
  const r1 = (v: number) => Math.round(v * 10) / 10;
  return {
    left: CYCLE_X.map((x) => r1(f(x / 100, 0, 1))),
    right: CYCLE_X.map((x) => r1(f(x / 100, 0.025 * asym, 1 - 0.12 * asym))),
  };
}

export function makeCurves(asym: number, skip: Joint[] = []): Partial<Record<Joint, JointCurve>> {
  const out: Partial<Record<Joint, JointCurve>> = {};
  for (const j of JOINTS) if (!skip.includes(j)) out[j] = makeCurve(j, asym);
  return out;
}

// ── 예시 대상자와 기록 ───────────────────────

export const SEED_SUBJECTS: Subject[] = [
  { id: 's1', name: '김영희', sex: '여', birth: '1947-05-14', heightCm: 152 },
  { id: 's2', name: '박정수', sex: '남', birth: '1944-02-03', heightCm: 168 },
  { id: 's3', name: '이순자', sex: '여', birth: '1951-01-20', heightCm: 149 },
  { id: 's4', name: '최민호', sex: '남', birth: '1956-03-11', heightCm: 171 },
  { id: 's5', name: '정옥순', sex: '여', birth: '1940-04-08', heightCm: 146 },
  { id: 's6', name: '한동길', sex: '남', birth: '1948-01-25', heightCm: 165 },
];

type Seed = Omit<ResultRecord, 'id' | 'curves' | 'ageAtCapture' | 'fileName'> & { asym: number };

function record(id: string, s: Seed): ResultRecord {
  const subject = SEED_SUBJECTS.find((x) => x.id === s.subjectId)!;
  const skip = (Object.keys(s.missing) as MissingKey[]).filter((k): k is Joint => k !== 'symmetry');
  const { asym, ...rest } = s;
  return {
    ...rest,
    id,
    ageAtCapture: ageOn(subject.birth, new Date(s.measuredAt)),
    curves: makeCurves(asym, skip),
    fileName: `walk_${fmtCompact(s.measuredAt)}.mp4`,
  };
}

export const SEED_RECORDS: ResultRecord[] = [
  record('GA-20250610-0142', {
    subjectId: 's1', measuredAt: '2025-06-10T14:14:00', heightAtCapture: 152,
    gdi: 82.4, cadence: 96, symmetry: 8.1, missing: {}, asym: 1,
  }),
  record('GA-20250318-0098', {
    subjectId: 's1', measuredAt: '2025-03-18T10:40:00', heightAtCapture: 152.5,
    gdi: 79.6, cadence: 92, symmetry: 10.3, missing: {}, asym: 1.3,
  }),
  record('GA-20241205-0311', {
    subjectId: 's1', measuredAt: '2024-12-05T15:05:00', heightAtCapture: 153,
    gdi: 80.1, cadence: 94, symmetry: null, asym: 1.1,
    missing: {
      symmetry: '영상 일부 구간에서 한쪽 다리가 가려져 계산할 수 없었습니다.',
      ankle: '영상에서 발목 부위가 충분히 보이지 않아 데이터를 산출하지 못했습니다.',
    },
  }),
  record('GA-20240902-0207', {
    subjectId: 's1', measuredAt: '2024-09-02T11:20:00', heightAtCapture: 153,
    gdi: 78.8, cadence: 90, symmetry: 11.2, missing: {}, asym: 1.4,
  }),
  record('GA-20250527-0130', {
    subjectId: 's3', measuredAt: '2025-05-27T09:30:00', heightAtCapture: 149,
    gdi: 85.2, cadence: 101, symmetry: 6.4, missing: {}, asym: 0.8,
  }),
  record('GA-20250402-0071', {
    subjectId: 's4', measuredAt: '2025-04-02T16:10:00', heightAtCapture: 171,
    gdi: 88.9, cadence: 104, symmetry: 5.2, missing: {}, asym: 0.6,
  }),
  record('GA-20250603-0125', {
    subjectId: 's5', measuredAt: '2025-06-03T13:45:00', heightAtCapture: 146,
    gdi: 74.3, cadence: 84, symmetry: 13.7, missing: {}, asym: 1.7,
  }),
];

export const MISSING_LABEL: Record<MissingKey, string> = {
  symmetry: '좌우 대칭 지표',
  hip: '고관절 그래프',
  knee: '무릎 그래프',
  ankle: '발목 그래프',
};
