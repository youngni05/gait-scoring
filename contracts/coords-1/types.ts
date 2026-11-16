/** Original MediaPipe values; z is relative depth, not metres. */
export type Landmark = readonly [x: number, y: number, z: number, visibility: number];
/** Exactly 33 landmarks at runtime; use validateCoords before consuming unknown data. */
export type PoseFrame = { readonly t: number; readonly lm: readonly Landmark[] | null };
export type StandMarks =
  | { readonly stand_start: number; readonly stand_end: number }
  | { readonly stand_start: null; readonly stand_end: null };
export interface Coords1 {
  readonly schema_version: 'coords-1';
  readonly recording_id: string;
  readonly subject: { readonly id: string; readonly height_cm: number };
  readonly capture: {
    readonly mode: 'guided' | 'fixed';
    readonly recorded_at: string;
    readonly width: number;
    readonly height: number;
    readonly orientation: 'landscape' | 'portrait';
    readonly device?: string;
  };
  readonly pose: {
    readonly model: 'pose_landmarker_lite' | 'pose_landmarker_full' | 'pose_landmarker_heavy';
    readonly engine_version: string;
    readonly delegate?: 'CPU' | 'GPU';
  };
  readonly marks: StandMarks;
  readonly frames: readonly PoseFrame[];
}
