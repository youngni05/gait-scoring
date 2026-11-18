/** UI uses Korean calendar dates; the existing wire contract keeps an explicit timezone. */
export function koreanMetadataTime(date: Date | null | undefined): string | null {
  if (!(date instanceof Date) || !Number.isFinite(date.getTime()) || date.getUTCFullYear() < 1970) return null;
  const local = new Date(date.getTime() + 9 * 60 * 60 * 1000);
  return local.toISOString().slice(0, 23) + '+09:00';
}
export function resolveRecordedAt(day: string, metadata: string | null) {
  if (!day) return { recordedAt: metadata ?? '', recordedAtSource: 'metadata' as const };
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  const parsed = new Date(day + 'T00:00:00Z');
  const valid = match && Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === day;
  return { recordedAt: valid ? day + 'T00:00:00+09:00' : '', recordedAtSource: 'user_entered' as const };
}
