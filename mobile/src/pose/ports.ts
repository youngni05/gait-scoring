import type { Coords1, PoseFrame, StandMarks } from '../../../contracts/coords-1/types';
import type { ValidationResult } from '../../../contracts/coords-1/validate';

export interface ProcessingInput {
  subjectId: string;
  heightCm: number;
  recordedAt: string;
  recordedAtSource: 'metadata' | 'user_entered';
  recordingId: string;
}
export interface VideoInfo {
  width: number;
  height: number;
  durationSeconds: number;
  orientation: 'landscape' | 'portrait';
  rotationAppliedDegrees: number;
}
/** Adapter-owned frame resource; opaque to contracts, UI and authentication. */
export interface DecodedFrame {
  t: number;
  pixels: unknown;
  release(): void;
}
export interface VideoSession {
  info: VideoInfo;
  frames(signal: AbortSignal): AsyncIterable<DecodedFrame>;
  dispose(): Promise<void>;
}
export interface VideoSource {
  open(localSelection: unknown, signal: AbortSignal): Promise<VideoSession>;
}
export interface PoseExtractor {
  readonly configuration: Coords1['pose'];
  extract(frame: DecodedFrame, signal: AbortSignal): Promise<PoseFrame>;
  dispose(): Promise<void>;
}
export type StandDetection =
  | { status: 'detected'; marks: { stand_start: number; stand_end: number }; version: string }
  | { status: 'not_detected'; marks: { stand_start: null; stand_end: null }; version: string; reason: string };
export interface StandDetector {
  detect(frames: readonly PoseFrame[], durationSeconds: number, signal: AbortSignal): Promise<StandDetection>;
}
/** Local metadata: not approved API fields. Never merge into coords-1. */
export interface ProcessingMetadata {
  recordedAtSource: ProcessingInput['recordedAtSource'];
  rotationAppliedDegrees: number;
  detection: StandDetection;
  modelAssetVersion?: string;
}
export interface SerializationInput {
  input: ProcessingInput;
  video: VideoInfo;
  pose: Coords1['pose'];
  frames: readonly PoseFrame[];
  marks: StandMarks;
}
export interface CoordsSerializer {
  /** Stage 6 implementation: guided only, <=5 coordinate decimals, original time precision. */
  create(input: SerializationInput): ValidationResult;
  stringify(validated: Coords1): string;
}
/** Internal adapter port, not an agreed HTTP request/response contract. */
export interface UploadGateway {
  submit(request: {
    coords: Coords1;
    metadata: ProcessingMetadata;
    requestId: string;
  }, signal: AbortSignal): Promise<{ receiptId: string; environment: 'mock' | 'server' }>;
}
