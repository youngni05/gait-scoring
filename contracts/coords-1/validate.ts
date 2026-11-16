import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import schema from './schema.json';
import type { Coords1 } from './types';

const ajv = new Ajv({ strict: true, allErrors: true, coerceTypes: false });
addFormats(ajv);
const validateShape = ajv.compile<Coords1>(schema);
export interface ContractIssue { path: string; message: string }
export type ValidationResult =
  | { ok: true; value: Coords1 }
  | { ok: false; issues: ContractIssue[] };
export interface ValidationContext {
  /** Optional decoded source duration. Kept outside the coords-1 body. */
  durationSeconds?: number;
  /** Injected by the caller; this comparison does not authenticate the subject. */
  expectedSubjectId?: string;
}

/** No mutation, coercion, clipping, interpolation or serialization. */
export function validateCoords(input: unknown, context: ValidationContext = {}): ValidationResult {
  if (!validateShape(input)) {
    return { ok: false, issues: (validateShape.errors ?? []).map(e => ({
      path: e.instancePath || '/', message: e.message ?? 'Invalid schema value',
    })) };
  }
  const issues: ContractIssue[] = [];
  const issue = (path: string, message: string) => issues.push({ path, message });
  input.frames.forEach((frame, i) => {
    const previous = input.frames[i - 1];
    if (previous && frame.t <= previous.t) issue(`/frames/${i}/t`, 'Time must strictly increase');
  });
  const last = input.frames[input.frames.length - 1]!.t;
  const duration = context.durationSeconds;
  if (duration !== undefined && (!Number.isFinite(duration) || duration < 0 || last > duration)) {
    issue('/context/durationSeconds', 'Duration must be finite, nonnegative and contain all frames');
  }
  const { stand_start: start, stand_end: end } = input.marks;
  // Without decoded duration, only the observed time coverage can be guaranteed.
  const upperBound = duration ?? last;
  if (start !== null && end !== null && (start >= end || end > upperBound)) {
    issue('/marks', 'Standing interval must have start < end and lie within source time coverage');
  }
  if (context.expectedSubjectId !== undefined && input.subject.id !== context.expectedSubjectId) {
    issue('/subject/id', 'Subject does not match caller context');
  }
  return issues.length ? { ok: false, issues } : { ok: true, value: input };
}
