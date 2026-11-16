/** Metadata only: never retains pixel buffers. Rejects incomplete/ambiguous timelines. */
export class CompleteFrameAudit {
  readonly expected: number[];
  count = 0;
  complete = false;
  constructor(timestamps: number[]) {
    this.expected = [...timestamps].sort((a, b) => a - b);
    if (!this.expected.length || this.expected.some((t, i, a) => !Number.isFinite(t) || t < 0 || (i > 0 && t <= a[i - 1]!)))
      throw new Error('비어 있거나 중복/음수 시각이 있는 영상은 지원하지 않습니다.');
  }
  accept(decodedTimestamp: number): number {
    const t = this.expected[this.count];
    if (t === undefined || !Number.isFinite(decodedTimestamp) || Math.abs(t - decodedTimestamp) > 0.0000011)
      throw new Error(`프레임 ${this.count + 1}: 원본 시각과 디코더 출력이 일치하지 않습니다.`);
    this.count++;
    return t;
  }
  finish(): void {
    if (this.count !== this.expected.length) throw new Error(`프레임 누락: ${this.count}/${this.expected.length}`);
    this.complete = true;
  }
}
