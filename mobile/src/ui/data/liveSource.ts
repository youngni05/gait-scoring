import { ApiError, errorMessage, type Participant } from '../../api/client';
import { extractCapture } from './analysis';
import { teamApi } from './teamApi';
import { readPending, keepPending, clearPending } from './pending-capture';
import type { Subject, ResultRecord } from './mock';
import { LoginFailedError, type GaitDataSource, type SubjectInput, type SubjectSummary } from './source';
import { visitKneeSummary } from './report-values.mjs';
import { walkingEstimate } from './visit-walking.mjs';

function subject(p: Participant): Subject {
  return {id:p.id, name:p.name, sex:p.gender === 'F' ? '여' : p.gender === 'M' ? '남' : null,
    birth:p.birth_date ?? '', heightCm:p.height_cm == null ? null : Number(p.height_cm)};
}
function profile(input: SubjectInput) {
  if (!input.sex || !input.birth || input.heightCm == null) throw Error('성별·생년월일·키를 입력해 주세요.');
  return {name:input.name, gender:input.sex === '여' ? 'F' as const : 'M' as const,
    birth_date:input.birth, height_cm:String(input.heightCm)};
}
const capturePath = (id:string) => '/participants/' + encodeURIComponent(id) + '/recordings';
const getRecord = (id:string, signal?:AbortSignal) => teamApi.shared<ResultRecord>('/recordings/'+encodeURIComponent(id), {}, signal);
// Partition retries by API and account; never save credentials in the outbox.
function pendingKey(id: string): string {
  const token = sessionStorage.getItem('team-access-token');
  if (!token) throw Error('다시 로그인한 뒤 저장 요청을 복구해 주세요.');
  const part = token.split('.')[1]!;
  const user = JSON.parse(atob(part.replace(/-/g, '+').replace(/_/g, '/'))).sub;
  if (typeof user !== 'string' || !user) throw Error('로그인 정보를 확인할 수 없습니다.');
  return JSON.stringify([teamApi.base, user, id]);
}

export const liveSource:GaitDataSource = {
  async login(username,password) {
    try { sessionStorage.setItem('team-access-token',await teamApi.login(username,password)); }
    catch (error) { if (error instanceof ApiError && error.status === 401) throw new LoginFailedError(); throw error; }
  },
  logout() { sessionStorage.removeItem('team-access-token'); },
  async pendingCapture(id) { const p = await readPending(pendingKey(id)); return p ? {id:p.id, fileName:p.fileName} : null; },
  listSubjects(signal) { return teamApi.shared<SubjectSummary[]>('/participant-summaries',{},signal); },
  async getSubject(id,signal) {
    try { return subject(await teamApi.participant(id,signal)); }
    catch (error) { if (error instanceof ApiError && error.status === 404) return null; throw error; }
  },
  async createSubject(input) { return subject(await teamApi.createParticipant(profile(input))); },
  async updateSubject(id,input) { return subject(await teamApi.updateParticipant(id,profile(input))); },
  async listResults(id,signal) {
    const rows: ResultRecord[] = [];
    for (;;) {
      const page = await teamApi.shared<ResultRecord[]>(capturePath(id)+'?offset='+rows.length, {}, signal);
      rows.push(...page);
      if (page.length < 20) return rows;
    }
  },
  async getResult(id,rid,signal) {
    try { const r = await getRecord(rid,signal); return r.subjectId === id ? r : null; }
    catch (error) { if (error instanceof ApiError && error.status === 404) return null; throw error; }
  },
  async deleteResult(id,rid) { await teamApi.shared(capturePath(id)+'/'+encodeURIComponent(rid),{method:'DELETE'}); },
  async analyzeVideo(id,file,progress,signal,capture) {
    const key = pendingKey(id);
    const check = () => {
      signal?.throwIfAborted();
      if (pendingKey(id) !== key) throw Error('원래 계정으로 로그인한 뒤 저장 요청을 복구해 주세요.');
    };
    let pending = await readPending(key);
    check();
    if (!pending) {
      if (!file) throw Error('복구할 요청이 없습니다. 영상을 선택해 주세요.');
      const packet = await extractCapture(file,id,capture?.day ?? '',capture?.mode ?? 'guided',
        signal ?? new AbortController().signal,progress,capture?.height ?? null,capture?.daySource);
      check();
      pending = await keepPending({key, id:packet.requestId, subjectId:id, fileName:file.name,
        body:JSON.stringify({fileName:file.name,packet})});
    }
    check();
    progress(95,'이전에 저장됐는지 확인하고 같은 요청으로 이어갑니다.');
    try {
      let row: ResultRecord;
      try { row = await getRecord(pending.id,signal); }
      catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error;
        check();
        const receipt = await teamApi.shared<{id:string;participant_id:string}>(capturePath(id), {
          method:'POST',headers:{'Content-Type':'application/json'},body:pending.body,
        },signal);
        if (receipt.id !== pending.id || receipt.participant_id !== id) throw Error('촬영 저장 번호가 일치하지 않습니다.');
        row = await getRecord(receipt.id,signal);
      }
      check();
      if (row.id !== pending.id || row.subjectId !== id) throw Error('촬영 저장 번호가 일치하지 않습니다.');
      await clearPending(key,pending.id);
      return row;
    } catch (error) { throw Error(errorMessage(error)); }
  },
  async downloadResult(id) {
    const r = await getRecord(id);
    const report = r.live ? {...r, walking_estimate:walkingEstimate(r.live), knee_observation_summary: {
      unit:'deg', time_unit:'ms', observations:visitKneeSummary(r.live)
    }} : r;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='gait-'+id+'.json';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
};
