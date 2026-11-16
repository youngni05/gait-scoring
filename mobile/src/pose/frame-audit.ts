export interface FrameSummary {
  count: number; first: number | null; last: number | null;
  maxGap: number; missedCallbacks: number; recent: number[];
}
export class FrameAudit {
  private previousPresented: number | null = null;
  private value: FrameSummary = { count: 0, first: null, last: null, maxGap: 0, missedCallbacks: 0, recent: [] };
  add(t: number, presented: number): void {
    if (!Number.isFinite(t) || t < 0 || !Number.isSafeInteger(presented) || presented < 0) throw new Error('비정상 프레임 시각/번호');
    const v = this.value;
    if (v.last !== null && t <= v.last) throw new Error('프레임 시각이 엄격하게 증가하지 않습니다.');
    if (this.previousPresented !== null && presented <= this.previousPresented) throw new Error('프레임 표시 번호가 증가하지 않습니다.');
    if (v.last !== null) v.maxGap = Math.max(v.maxGap, t - v.last);
    if (this.previousPresented !== null) v.missedCallbacks += Math.max(0, presented - this.previousPresented - 1);
    this.previousPresented = presented;
    v.first ??= t; v.last = t; v.count++;
    v.recent.push(t); if (v.recent.length > 10) v.recent.shift();
  }
  snapshot(): FrameSummary { return { ...this.value, recent: [...this.value.recent] }; }
}
