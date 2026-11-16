import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import schema from '../../../contracts/coords-1/schema.json';
import type { ProcessingInput } from './ports';
const ajv = new Ajv({ strict: true });
addFormats(ajv);
const dateValid = ajv.compile(schema.properties.capture.properties.recorded_at);
export function checkProcessingInput(input: Omit<ProcessingInput, 'recordingId'>): string[] {
  const errors: string[] = [];
  if (!input.subjectId.trim()) errors.push('시험용 대상자 ID를 입력하세요.');
  if (!Number.isFinite(input.heightCm) || input.heightCm < 100 || input.heightCm > 220) errors.push('키는 100~220cm 숫자로 입력하세요.');
  if (!dateValid(input.recordedAt)) errors.push('촬영일은 시간대를 포함한 형식으로 입력하세요. 예: 2026-11-16T10:00:00+09:00');
  if (!['metadata', 'user_entered'].includes(input.recordedAtSource)) errors.push('촬영일 출처를 선택하세요.');
  return errors;
}
