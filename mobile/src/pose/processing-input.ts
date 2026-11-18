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
  if (!dateValid(input.recordedAt)) errors.push('유효한 촬영일이 없습니다. 영상 메타데이터를 확인하거나 촬영일을 년·월·일로 입력하세요.');
  if (!['metadata', 'user_entered'].includes(input.recordedAtSource)) errors.push('촬영일 출처를 선택하세요.');
  return errors;
}
