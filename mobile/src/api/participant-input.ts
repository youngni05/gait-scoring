import type { Participant } from './client';

export interface ParticipantInput { name: string; gender: string; birth_date: string; height_cm: string }
export interface ParticipantBody { name: string; gender: 'M' | 'F'; birth_date: string; height_cm: string }
export type ParticipantErrors = Partial<Record<keyof ParticipantInput, string>>;
export function koreaToday(now = new Date()): string {
  return new Date(now.getTime() + 9 * 60 * 60 * 1000).toISOString().slice(0, 10);
}
export function validateParticipant(input: ParticipantInput, today = koreaToday()): { body: ParticipantBody | null; errors: ParticipantErrors } {
  const errors: ParticipantErrors = {};
  const name = input.name.trim(), day = input.birth_date;
  if (!name || [...name].length > 100) errors.name = '이름을 1~100자로 입력하세요.';
  if (input.gender !== 'M' && input.gender !== 'F') errors.gender = '성별을 선택하세요.';
  const date = new Date(day + 'T00:00:00Z');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || day.startsWith('0000') || !Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== day || day > today) errors.birth_date = '오늘 이전 또는 오늘의 실제 날짜를 입력하세요.';
  const height = input.height_cm.trim();
  if (!/^\d{1,3}(\.\d{1,2})?$/.test(height) || Number(height) <= 0 || Number(height) > 999.99) errors.height_cm = '키는 0보다 크고 999.99 이하인 숫자로, 소수 둘째 자리까지 입력하세요.';
  if (Object.keys(errors).length) return { body: null, errors };
  return { body: { name, gender: input.gender as 'M' | 'F', birth_date: day, height_cm: Number(height).toFixed(2) }, errors };
}
/** PATCH only changed profile fields; never send measurement data or identity. */
export function participantChanges(original: Participant, body: ParticipantBody): Partial<ParticipantBody> {
  const patch: Partial<ParticipantBody> = {};
  if (body.name !== original.name) patch.name = body.name;
  if (body.gender !== original.gender) patch.gender = body.gender;
  if (body.birth_date !== original.birth_date) patch.birth_date = body.birth_date;
  if (original.height_cm === null || Number(body.height_cm) !== Number(original.height_cm)) patch.height_cm = body.height_cm;
  return patch;
}
