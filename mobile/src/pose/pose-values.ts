import type { PoseFrame, Landmark } from '../../../contracts/coords-1/types';
export interface RawLandmark { x: number; y: number; z: number; visibility?: number }
/** Preserve all original values: no rounding, clamping, interpolation or mirroring. */
export function toPoseFrame(t: number, poses: readonly (readonly RawLandmark[])[]): PoseFrame {
  if (!Number.isFinite(t) || t < 0) throw new Error('비정상 원본 시각');
  if (poses.length === 0) return { t, lm: null };
  if (poses.length !== 1 || poses[0]!.length !== 33) throw new Error('단일 대상 33관절 출력이 아닙니다.');
  const lm: Landmark[] = poses[0]!.map(p => {
    if (![p.x, p.y, p.z, p.visibility].every(v => typeof v === 'number' && Number.isFinite(v)) || p.visibility! < 0 || p.visibility! > 1)
      throw new Error('비정상 관절 원본값');
    return [p.x, p.y, p.z, p.visibility!];
  });
  return { t, lm };
}
