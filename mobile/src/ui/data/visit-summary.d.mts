import type { visitKneeSummary } from './report-values.mjs';
import type { walkingEstimate } from './visit-walking.mjs';

type KneeRow = ReturnType<typeof visitKneeSummary>[number];
export interface DirectionGdi {
  direction: string; left: number; right: number; total_ms: number;
  segments: { segmentId: string; start: number; stop: number; weight: number; share: number; left: number; right: number }[];
}
export interface VisitGdi { value: number | null; left: number | null; right: number | null; directions: string[]; byDirection: DirectionGdi[] }
export interface VisitSummary {
  directions: string[]; available: string[]; gdi: VisitGdi; walking: ReturnType<typeof walkingEstimate>;
  gap: { value: number | null; count: number; segments: number };
  knee: Partial<Record<'left' | 'right', KneeRow>>; kneeRows: KneeRow[];
}
export type Verdict = 'better' | 'worse' | 'similar';
export interface CompareRow {
  key: string; label: string; unit: string; digits: number; band: number;
  current: number; previous: number; delta: number; verdict: Verdict; meaning: string;
}
export interface Comparison { previousId: string; previousDate: string; directions: string[]; skipped: string[]; notes: string[]; rows: CompareRow[] }
export interface Trend extends CompareRow {
  points: { date: string; value: number; current?: boolean }[]; steady: boolean;
  since: { date: string; delta: number; verdict: Verdict; directions: string[] } | null;
}

export const DIRECTIONS: string[];
export const EXAMPLE_BAND: { gdi: number; cadence: number; knee: number; stepGap: number };
export function visitDirections(visit: unknown): string[];
export function directionGdi(visit: unknown, direction: string): DirectionGdi | null;
export function visitGdi(visit: unknown, directions?: string[]): VisitGdi;
export function stepGap(walking: ReturnType<typeof walkingEstimate>): VisitSummary['gap'];
export function visitSummary(visit: unknown, directions?: string[]): VisitSummary;
export function describeVisit(summary: VisitSummary): string[];
export function compareVisits(current: unknown, prior: unknown): Comparison | null;
export function visitHistory(current: unknown, priors: unknown[]): {
  recent: Comparison | null; trends: Trend[]; lines: string[]; visits: number; comparable?: number; sentence: string;
};
export const fmtDay: (iso: string | null | undefined) => string;
