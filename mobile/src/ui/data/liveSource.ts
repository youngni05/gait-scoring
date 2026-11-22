import { ApiError, errorMessage, type Participant } from '../../api/client';
import { extractCapture } from './analysis';
import { teamApi } from './teamApi';
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
// Keep only the current submission for this selected File so a lost receipt can replay identical bytes.
let submissions = new WeakMap<File, {settings:string; body:string; id:string}>();

export const liveSource:GaitDataSource = {
  async login(username,password) {
    try { sessionStorage.setItem('team-access-token',await teamApi.login(username,password)); }
    catch (error) { if (error instanceof ApiError && error.status === 401) throw new LoginFailedError(); throw error; }
  },
  logout() { sessionStorage.removeItem('team-access-token'); submissions = new WeakMap(); },
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
    if (!file) throw Error('MP4 영상을 선택해 주세요.');
    const settings = JSON.stringify([id, capture]);
    let pending = submissions.get(file);
    if (!pending || pending.settings !== settings) {
      const packet = await extractCapture(file,id,capture?.day ?? '',capture?.mode ?? 'guided',
        signal ?? new AbortController().signal,progress,capture?.height ?? null,capture?.daySource);
      pending = {settings, body:JSON.stringify({fileName:file.name,packet}), id:packet.requestId};
      submissions.set(file,pending);
    }
    progress(95,'팀 서버에 촬영 한 건의 좌표를 저장하고 있습니다.');
    try {
      const receipt = await teamApi.shared<{id:string;participant_id:string}>(capturePath(id), {
        method:'POST',headers:{'Content-Type':'application/json'},body:pending.body,
      },signal);
      if (receipt.id !== pending.id || receipt.participant_id !== id) throw Error('촬영 저장 번호가 일치하지 않습니다.');
      const row = await getRecord(receipt.id,signal);
      submissions.delete(file);
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
