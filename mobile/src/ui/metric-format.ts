/** 결과 목록과 상세 화면에서 같은 자릿수로 표시한다. */
export function formatMetric(value: number | null, digits: number): string {
  return value !== null && Number.isFinite(value) ? value.toFixed(digits) : '—';
}
