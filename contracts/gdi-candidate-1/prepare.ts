import { validateCoords } from '../coords-1/validate';

/** Anatomical sides, independent of camera mirroring. Foot index is NOT BODY25 big toe. */
const SIDES = { left: [27, 25, 23, 31], right: [28, 26, 24, 32] } as const;
export interface CandidatePoint { x: number; y: number; visibility: number }
export interface CandidateSide {
  ankle: CandidatePoint;
  knee: CandidatePoint;
  hip: CandidatePoint;
  footIndex: CandidatePoint;
}
export interface GdiCandidate {
  schema_version: 'gdi-candidate-1';
  readyForInference: false;
  recordingId: string;
  units: 'pixels';
  frames: Array<{ t: number; sides: { left: CandidateSide; right: CandidateSide } | null }>;
  blockers: readonly string[];
}

/** Inspection-only intermediate data, never a frozen model tensor or a quality approval.
 * Preserves frames/PTS/nulls/visibility; does not choose a gait interval or impute data.
 */
export function prepareGdiCandidate(input: unknown): GdiCandidate {
  const checked = validateCoords(input);
  if (!checked.ok) throw new Error('Invalid coords-1: ' + checked.issues.map(i => `${i.path}: ${i.message}`).join('; '));
  const coords = checked.value;
  return {
    schema_version: 'gdi-candidate-1', readyForInference: false,
    recordingId: coords.recording_id, units: 'pixels',
    frames: coords.frames.map(frame => {
      if (frame.lm === null) return { t: frame.t, sides: null };
      const side = (indices: readonly number[]): CandidateSide => {
        const point = (index: number): CandidatePoint => {
          const p = frame.lm![index]!;
          return { x: p[0] * coords.capture.width, y: p[1] * coords.capture.height, visibility: p[3] };
        };
        return { ankle: point(indices[0]!), knee: point(indices[1]!), hip: point(indices[2]!), footIndex: point(indices[3]!) };
      };
      return { t: frame.t, sides: { left: side(SIDES.left), right: side(SIDES.right) } };
    }),
    blockers: [
      'foot_index_is_not_validated_as_body25_big_toe',
      'mediapipe_missingness_policy_not_validated',
      'gait_interval_and_frame_sampling_not_validated',
      'historical_preprocessing_parity_not_verified',
    ],
  };
}
