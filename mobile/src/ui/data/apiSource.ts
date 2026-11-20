import { ApiClient, ApiError, type Participant, type MeasurementResult, type MeasurementItem } from '../../api/client';
import type { PreparedMeasurement } from '../../../../contracts/measurement-1/prepare';
import { LoginFailedError, type GaitDataSource, type SubjectInput } from './source';
import { JOINTS, type Subject, type ResultRecord } from './mock';

export const AUTH_EXPIRED = 'gait-auth-expired';
export function mapSubject(p: Participant): Subject {
  return { id: p.id, name: p.name, sex: p.gender === 'F' ? '여' : p.gender === 'M' ? '남' : null,
    birth: p.birth_date ?? '', heightCm: p.height_cm === null ? null : Number(p.height_cm) };
}
export function mapResult(value: MeasurementResult): ResultRecord {
  const a = value.analysis_result;
  const missing: ResultRecord['missing'] = {};
  const curves: ResultRecord['curves'] = {};
  if (a) {
    if (a.gdi_score === null) missing.gdi = '서버에 저장된 값이 없습니다.';
    if (a.cadence_spm === null) missing.cadence = '서버에 저장된 값이 없습니다.';
    // Do not reinterpret arbitrary graph JSON as normalized gait-cycle angles.
    for (const joint of JOINTS) {
      const graph = a[`${joint}_graph_json`] as { schema_version?: string; left?: unknown; right?: unknown } | null;
      if (graph?.schema_version === 'gait-cycle-degrees-1' &&
          Array.isArray(graph.left) && Array.isArray(graph.right) &&
          graph.left.length === 51 && graph.right.length === 51 &&
          [...graph.left, ...graph.right].every(v => typeof v === 'number' && Number.isFinite(v))) {
        curves[joint] = { left: graph.left as number[], right: graph.right as number[] };
      } else missing[joint] = graph == null ? '서버에 저장된 그래프가 없습니다.' : '그래프 축·단위가 확인되지 않아 표시하지 않습니다.';
    }
  }
  return { id: value.measurement.id, subjectId: value.measurement.participant_id,
    measuredAt: value.measurement.analysis_started_at ?? value.measurement.measured_at ?? '', ageAtCapture: value.measurement.age,
    dateBasis: value.measurement.analysis_started_at ? 'analysis_start' : 'legacy_capture',
    heightAtCapture: value.participant.height_cm === null ? null : Number(value.participant.height_cm),
    gdi: a?.gdi_score ?? null, cadence: a?.cadence_spm ?? null, strideLength: a?.stride_length_cm ?? null,
    symmetry: null, curves, missing, fileName: '', status: value.status };
}
const body = (s: SubjectInput) => {
  if (!s.sex || !s.birth || s.heightCm === null) throw new Error('성별·생년월일·키를 입력하세요.');
  return { name: s.name, gender: s.sex === '여' ? 'F' as const : 'M' as const, birth_date: s.birth, height_cm: String(s.heightCm) };
};

/** Token stays in memory. A session renewal must use the original account. */
export class ApiDataSource implements GaitDataSource {
  private token = '';
  private username = '';
  private client: ApiClient;
  constructor(base: string, transport?: typeof fetch) { this.client = new ApiClient(base, () => this.token, transport); }
  get base() { return this.client.base; }
  configure(base: string) {
    if (this.token) throw new Error('서버를 바꾸려면 먼저 로그아웃하세요.');
    this.client = new ApiClient(base, () => this.token);
  }
  async login(username: string, password: string) {
    if (this.username && username !== this.username) throw new Error('진행 중인 요청을 유지하려면 같은 계정으로 로그인하세요.');
    try { this.token = await this.client.login(username, password); this.username = username; }
    catch (e) { if (e instanceof ApiError && e.status === 401) throw new LoginFailedError(); throw e; }
  }
  logout() { this.token = ''; this.username = ''; }
  private async authenticated<T>(request: () => Promise<T>): Promise<T> {
    try { return await request(); }
    catch (e) {
      if (e instanceof ApiError && e.status === 401) window.dispatchEvent(new Event(AUTH_EXPIRED));
      throw e;
    }
  }
  async allMeasurements(id: string, signal?: AbortSignal): Promise<MeasurementItem[]> {
    const rows: MeasurementItem[] = [];
    for (let offset = 0; ; offset += 20) {
      const page = await this.authenticated(() => this.client.measurements(id, offset, signal));
      rows.push(...page);
      if (page.length < 20) return rows;
    }
  }
  async listSubjects(signal?: AbortSignal) {
    const people = await this.authenticated(() => this.client.participants(signal));
    const summaries = [];
    // Bound concurrency; no result/graph downloads are needed for this screen.
    for (let i = 0; i < people.length; i += 4) {
      summaries.push(...await Promise.all(people.slice(i, i + 4).map(async p => {
        const rows = await this.allMeasurements(p.id, signal);
        return { ...mapSubject(p), latestMeasuredAt: rows[0]?.analysis_started_at ?? rows[0]?.measured_at ?? null, resultCount: rows.length };
      })));
    }
    return summaries;
  }
  async getSubject(id: string, signal?: AbortSignal) {
    try { return mapSubject(await this.authenticated(() => this.client.participant(id, signal))); }
    catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e; }
  }
  async createSubject(input: SubjectInput) { return mapSubject(await this.authenticated(() => this.client.createParticipant(body(input)))); }
  async updateSubject(id: string, input: SubjectInput) { return mapSubject(await this.authenticated(() => this.client.updateParticipant(id, body(input)))); }
  async listResults(id: string, signal?: AbortSignal) {
    const rows = await this.allMeasurements(id, signal);
    const results: ResultRecord[] = [];
    for (let i = 0; i < rows.length; i += 4) {
      const page = await Promise.all(rows.slice(i, i + 4).map(row => this.getResult(id, row.id, signal)));
      results.push(...page.filter((r): r is ResultRecord => r !== null));
    }
    return results;
  }
  async getResult(subjectId: string, id: string, signal?: AbortSignal) {
    try {
      const result = await this.authenticated(() => this.client.result(id, signal));
      return result.measurement.participant_id === subjectId ? mapResult(result) : null;
    } catch (e) { if (e instanceof ApiError && e.status === 404) return null; throw e; }
  }
  saveMeasurement(request: PreparedMeasurement, signal?: AbortSignal) {
    return this.authenticated(() => this.client.save(request, signal));
  }
  async analyzeVideo(): Promise<ResultRecord> { throw new Error('자동 AI 분석은 아직 연결되지 않았습니다. 측정 저장 화면을 이용하세요.'); }
  async downloadResult(id: string, signal?: AbortSignal) {
    const result = await this.authenticated(() => this.client.result(id, signal));
    if (signal?.aborted) throw new DOMException('취소됨', 'AbortError');
    if (result.status !== 'available') throw new Error('아직 내려받을 분석 결과가 없습니다.');
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `gait-result-${id}.json`;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
