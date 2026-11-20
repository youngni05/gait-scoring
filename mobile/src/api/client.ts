import type { PreparedMeasurement } from '../../../contracts/measurement-1/prepare';
import type { ParticipantBody } from './participant-input';

export interface Participant { id: string; name: string; birth_date: string | null; height_cm: string | number | null; gender: 'M' | 'F' | null }
export interface Measurement { id: string; participant_id: string; age: number; measured_at: string | null; analysis_started_at?: string | null; extractor_name: string; extractor_version: string; created_by: string }
export interface MeasurementItem extends Measurement { has_analysis_result: boolean }
export interface MeasurementResult {
  status: 'available' | 'not_available'; participant: Participant; measurement: Measurement;
  analysis_result: null | { gdi_score: number | null; stride_length_cm: number | null; cadence_spm: number | null; hip_graph_json?: unknown; knee_graph_json?: unknown; ankle_graph_json?: unknown; model_version: string; analyzed_at: string };
}
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public retryAfter = 0) { super(message); }
}
export function normalizeBase(value: string): string {
  const url = new URL(value.trim());
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('서버 주소 형식을 확인하세요.');
  if (url.protocol === 'http:' && !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) throw new Error('외부 서버는 HTTPS 주소를 사용하세요.');
  return url.href.replace(/\/$/, '');
}
export function errorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return error instanceof Error ? error.message : '요청을 완료하지 못했습니다.';
  if (error.status === 401) return '인증이 만료되었거나 로그인 정보가 잘못되었습니다. 같은 계정으로 로그인한 뒤 다시 시도하세요.';
  const messages: Record<string, string> = {
    IDEMPOTENCY_CONFLICT: '이 요청 키로 다른 내용이 저장되어 있습니다. 기록을 확인하세요.',
    REQUEST_IN_PROGRESS: '서버가 같은 요청을 처리 중입니다. 잠시 후 같은 요청으로 재시도하세요.',
    PARTICIPANT_BIRTH_DATE_REQUIRED: '대상자의 생년월일을 등록한 뒤 재시도하세요.',
    PARTICIPANT_MISMATCH: '선택한 대상자와 좌표의 대상자가 다릅니다. 입력을 확인하세요.',
    NETWORK: '서버 응답을 확인하지 못했습니다. 서버 주소·네트워크·CORS를 확인하세요. 저장 요청은 같은 키로 재시도하세요.',
    CANCELLED: '응답 대기를 취소했습니다. 이미 저장되었을 수 있으므로 같은 요청으로 재시도하세요.',
    INVALID_RESPONSE: '서버 응답 형식을 확인하지 못했습니다. 저장 요청은 같은 키로 재시도하세요.',
  };
  return messages[error.code] ?? (error.status === 422 ? '입력 검증에 실패했습니다. 촬영일·대상자 정보·좌표를 확인하세요.' : `요청 실패 (${error.status}). 다시 시도하기 전에 서버 상태를 확인하세요.`);
}
export class ApiClient {
  readonly base: string;
  // Native browser fetch needs the global receiver, not this ApiClient instance.
  constructor(base: string, private token: () => string, private transport: typeof fetch = (input, init) => globalThis.fetch(input, init), private timeoutMs = 30000) { this.base = normalizeBase(base); }
  private async request(path: string, init: RequestInit = {}, signal?: AbortSignal, authenticated = true): Promise<{ data: any; status: number }> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
    const timer = setTimeout(abort, this.timeoutMs);
    try {
      const headers = new Headers(init.headers);
      if (authenticated) headers.set('Authorization', `Bearer ${this.token()}`);
      const response = await this.transport(this.base + path, { ...init, headers, signal: controller.signal, credentials: 'omit', redirect: 'error', cache: 'no-store' });
      let data: any;
      try { data = await response.json(); } catch { throw new ApiError(response.status, 'INVALID_RESPONSE', 'Invalid JSON response'); }
      if (!response.ok) throw new ApiError(response.status, typeof data?.detail?.code === 'string' ? data.detail.code : 'HTTP_ERROR', 'Request failed', Math.max(0, Number(response.headers.get('Retry-After')) || 0));
      return { data, status: response.status };
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, signal?.aborted ? 'CANCELLED' : 'NETWORK', 'Response unavailable');
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
  }
  async login(username: string, password: string, signal?: AbortSignal): Promise<string> {
    const { data } = await this.request('/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ username, password }).toString() }, signal, false);
    if (typeof data?.access_token !== 'string' || !data.access_token || data.token_type?.toLowerCase() !== 'bearer') throw new ApiError(200, 'INVALID_RESPONSE', 'Invalid token response');
    return data.access_token;
  }
  async participants(signal?: AbortSignal): Promise<Participant[]> {
    const { data } = await this.request('/participants', {}, signal);
    if (!Array.isArray(data) || !data.every(p => typeof p.id === 'string' && typeof p.name === 'string')) throw new ApiError(200, 'INVALID_RESPONSE', 'Invalid participants');
    return data;
  }
  async participant(id: string, signal?: AbortSignal): Promise<Participant> {
    const { data } = await this.request(`/participants/${encodeURIComponent(id)}`, {}, signal);
    if (data?.id !== id || typeof data.name !== 'string') throw new ApiError(200, 'INVALID_RESPONSE', 'Invalid participant');
    return data;
  }
  async createParticipant(body: ParticipantBody, signal?: AbortSignal): Promise<Participant> {
    const { data, status } = await this.request('/participants', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, signal);
    return this.participantResponse(data, status, 201);
  }
  async updateParticipant(id: string, body: Partial<ParticipantBody>, signal?: AbortSignal): Promise<Participant> {
    if (!Object.keys(body).length) throw new Error('수정할 정보가 없습니다.');
    const { data, status } = await this.request(`/participants/${encodeURIComponent(id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, signal);
    const participant = this.participantResponse(data, status, 200);
    if (participant.id !== id) throw new ApiError(status, 'INVALID_RESPONSE', 'Wrong participant');
    return participant;
  }
  private participantResponse(data: any, status: number, expected: number): Participant {
    if (status !== expected || typeof data?.id !== 'string' || !/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(data.id) || typeof data.name !== 'string' || !['M', 'F'].includes(data.gender) || typeof data.birth_date !== 'string' || !['string', 'number'].includes(typeof data.height_cm) || !Number.isFinite(Number(data.height_cm)) || Number(data.height_cm) <= 0) throw new ApiError(status, 'INVALID_RESPONSE', 'Invalid participant response');
    return data;
  }
  async save(request: PreparedMeasurement, signal?: AbortSignal): Promise<Measurement> {
    const { data, status } = await this.request(request.path, { method: request.method, headers: request.headers, body: request.body }, signal);
    const participant = request.path.split('/')[2];
    if (![200, 201].includes(status) || typeof data?.id !== 'string' || !data.id || data.participant_id !== participant) throw new ApiError(status, 'INVALID_RESPONSE', 'Invalid measurement receipt');
    return data;
  }
  async measurements(participant: string, offset: number, signal?: AbortSignal): Promise<MeasurementItem[]> {
    const { data } = await this.request(`/participants/${encodeURIComponent(participant)}/measurements?limit=20&offset=${offset}`, {}, signal);
    if (!Array.isArray(data) || !data.every(m => typeof m.id === 'string' && m.participant_id === participant && typeof (m.analysis_started_at ?? m.measured_at) === 'string')) throw new ApiError(200, 'INVALID_RESPONSE', 'Invalid measurements');
    return data;
  }
  async result(id: string, signal?: AbortSignal): Promise<MeasurementResult> {
    const { data } = await this.request(`/measurements/${encodeURIComponent(id)}/result`, {}, signal);
    if (!['available', 'not_available'].includes(data?.status) || data?.measurement?.id !== id || (data.status === 'available' && !data.analysis_result) || (data.status === 'not_available' && data.analysis_result !== null)) throw new ApiError(200, 'INVALID_RESPONSE', 'Invalid result');
    return data;
  }
}
