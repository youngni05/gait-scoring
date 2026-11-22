/**
 * 가상 데이터 구현. 서버·분석 모델 없이 화면을 확인하기 위한 것이다.
 * 메모리에만 저장하므로 새로고침하면 예시 데이터로 돌아간다.
 */
import { getDemo } from './demo';
import { SEED_RECORDS, SEED_SUBJECTS, ageOn, fmtCompact, makeCurves, type ResultRecord, type Subject } from './mock';
import { AnalysisFailedError, LoginFailedError, type GaitDataSource, type SubjectSummary } from './source';

let subjects: Subject[] = [...SEED_SUBJECTS];
let records: ResultRecord[] = [...SEED_RECORDS];
let seq = 1;

/** 서버 응답을 기다리는 시간을 흉내 낸다. 취소 신호가 오면 바로 멈춘다. */
function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('취소됨', 'AbortError'));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new DOMException('취소됨', 'AbortError'));
    }, { once: true });
  });
}

const byLatest = (a: ResultRecord, b: ResultRecord) => b.measuredAt.localeCompare(a.measuredAt);
const round1 = (v: number) => Math.round(v * 10) / 10;

/** 분석 실패 데모에서 멈출 진행률. ‘보행 구간 찾기’ 단계에서 실패한 것으로 보여준다. */
const FAIL_AT = 38;
/** 가상 분석 전체 시간(ms) */
const RUN_MS = 6000;

export const mockSource: GaitDataSource = {
  async login() {
    await wait(500);
    if (getDemo().login === 'fail') throw new LoginFailedError('아이디 또는 비밀번호가 올바르지 않습니다.');
  },

  logout() {},
  async discardRejectedCapture() { return null; },
  async pendingCapture() { return null; },

  async listSubjects(signal) {
    await wait(600, signal);
    const demo = getDemo().subjectList;
    if (demo === 'loading') await wait(1e9, signal);
    if (demo === 'empty') return [];
    return subjects.map<SubjectSummary>((s) => {
      const mine = records.filter((r) => r.subjectId === s.id).sort(byLatest);
      return { ...s, latestMeasuredAt: mine[0]?.measuredAt ?? null, resultCount: mine.length };
    });
  },

  async getSubjectSummary(id, signal) {
    return (await this.listSubjects(signal)).find(s => s.id === id) ?? null;
  },
  async listResultPage(id, offset, signal) {
    const rows = await this.listResults(id, signal);
    return {items:rows.slice(offset,offset+20), total:rows.length, offset, limit:20};
  },
  async getSubject(id, signal) {
    await wait(150, signal);
    return subjects.find((s) => s.id === id) ?? null;
  },

  async createSubject(input) {
    await wait(400);
    const created = { ...input, id: `s${Date.now().toString(36)}${seq++}`, createdAt: new Date().toISOString() };
    subjects = [created, ...subjects];
    return created;
  },

  async updateSubject(id, input) {
    await wait(400);
    const updated = { ...input, id, createdAt: subjects.find(s => s.id === id)?.createdAt ?? null };
    subjects = subjects.map((s) => (s.id === id ? updated : s));
    return updated;
  },

  async listResults(subjectId, signal) {
    await wait(300, signal);
    return records.filter((r) => r.subjectId === subjectId).sort(byLatest);
  },

  async getResult(subjectId, resultId, signal) {
    await wait(200, signal);
    return records.find((r) => r.id === resultId && r.subjectId === subjectId) ?? null;
  },

  async deleteResult(subjectId, resultId) {
    records = records.filter(r => r.id !== resultId || r.subjectId !== subjectId);
  },

  async analyzeVideo(subjectId, video, onProgress, signal) {
    const outcome = getDemo().analysis; // 시작 시점의 데모 설정으로 끝까지 진행한다.
    const subject = subjects.find((s) => s.id === subjectId);
    if (!subject) throw new AnalysisFailedError('GA-404', '대상자를 찾을 수 없습니다.');
    const start = Date.now();
    for (;;) {
      await wait(100, signal);
      const p = Math.min(100, ((Date.now() - start) / RUN_MS) * 100);
      if (outcome === 'fail' && p >= FAIL_AT) {
        onProgress(FAIL_AT);
        throw new AnalysisFailedError('GA-102', '영상에서 보행 구간을 찾지 못했습니다');
      }
      onProgress(Math.floor(p));
      if (p >= 100) break;
    }
    const partial = outcome === 'partial';
    const now = new Date();
    const asym = 0.6 + Math.random() * 0.9;
    const rec: ResultRecord = {
      id: `GA-${fmtCompact(now.toISOString())}-${String(Math.floor(Math.random() * 9000) + 1000)}`,
      subjectId,
      measuredAt: now.toISOString(),
      ageAtCapture: ageOn(subject.birth, now),
      heightAtCapture: subject.heightCm,
      gdi: round1(76 + Math.random() * 12),
      cadence: Math.round(86 + Math.random() * 16),
      symmetry: partial ? null : round1(5 + asym * 5),
      curves: makeCurves(asym, partial ? ['ankle'] : []),
      missing: partial
        ? {
            symmetry: '영상 일부 구간에서 한쪽 다리가 가려져 계산할 수 없었습니다.',
            ankle: '영상에서 발목 부위가 충분히 보이지 않아 데이터를 산출하지 못했습니다.',
          }
        : {},
      fileName: video?.name ?? 'walk_sample.mp4',
    };
    records = [rec, ...records];
    return rec;
  },

  async downloadResult() {
    await wait(1800);
    if (getDemo().download === 'fail') throw new Error('휴대폰 저장 공간이 부족하거나 네트워크 연결이 불안정합니다.');
  },
};
