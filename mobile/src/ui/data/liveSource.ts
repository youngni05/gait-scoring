import { ApiError, errorMessage, type Participant } from '../../api/client';
import { extractCapture } from './analysis';
import { teamApi } from './teamApi';
import { readPending, keepPending, clearPending, markDelivery, withPendingLock, checkLease } from './pending-capture';
import type { Subject, ResultRecord } from './mock';
import { LoginFailedError, type GaitDataSource, type SubjectInput, type SubjectSummary, type ResultPage } from './source';
import { visitKneeSummary } from './report-values.mjs';
import { visitGdi } from './visit-summary.mjs';
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
const getRecord = async (id:string, signal?:AbortSignal): Promise<ResultRecord> => {
  const row = await teamApi.shared<ResultRecord>('/recordings/'+encodeURIComponent(id), {}, signal);
  if (!row || row.id !== id || typeof row.subjectId !== 'string' || !['pending','failed','ready','legacy'].includes(row.serverState ?? '') || !row.curves || !row.missing)
    throw new ApiError(200, 'INVALID_RESPONSE', '저장된 결과를 확인할 수 없습니다. 요청을 보관합니다.');
  return row;
};
const isMissing = (e: unknown) => e instanceof ApiError && e.status === 404 && e.code === 'HTTP_ERROR' && e.detail === 'Measurement not found';
const rejectionDetails = new Set(['Coordinate payload exceeds limit','Expected coordinate JSON','Expected a coordinate packet and file name','Invalid file name','Invalid coordinate packet','Recording or participant mismatch','Invalid capture date']);
const isRejected = (e: unknown): e is ApiError => e instanceof ApiError && [413,415,422].includes(e.status) && e.code === 'HTTP_ERROR' && typeof e.detail === 'string' && rejectionDetails.has(e.detail);
function checkPendingBody(body: string, id: string, subjectId: string) {
  let saved;
  try { saved = JSON.parse(body); } catch { throw Error('보관된 분석 정보를 읽을 수 없습니다. 데이터를 유지했으니 담당자에게 확인해 주세요.'); }
  if (saved?.packet?.requestId !== id || saved?.packet?.coords?.subject?.id !== subjectId || typeof saved.fileName !== 'string')
    throw Error('보관된 분석 정보가 맞지 않습니다. 데이터를 유지했으니 담당자에게 확인해 주세요.');
}

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
  async pendingCapture(id) { const p = await readPending(pendingKey(id)); return p ? {id:p.id, fileName:p.fileName, canDiscard: p.delivery === 'rejected', rejectionStatus:p.rejectionStatus} : null; },
  async discardRejectedCapture(id, requestId, signal) {
    const key = pendingKey(id);
    return withPendingLock(key, async (lease) => {
      const check = async () => { await checkLease(lease); signal?.throwIfAborted(); if (pendingKey(id) !== key) throw Error('같은 계정으로 다시 로그인해 주세요.'); };
      await check();
      const pending = await readPending(key);
      if (!pending || pending.id !== requestId || pending.delivery !== 'rejected')
        throw Error('정리할 수 있는 요청이 아닙니다. 저장 요청 상태를 다시 확인해 주세요.');
      let row: ResultRecord | null = null;
      try { row = await getRecord(requestId, signal); }
      catch (error) { if (!isMissing(error)) throw error; }
      await check();
      if (row && (row.id !== requestId || row.subjectId !== id)) throw Error('저장된 결과를 확인할 수 없습니다.');
      await clearPending(key, requestId, lease);
      return row;
    });
  },
  listSubjects(signal) { return teamApi.shared<SubjectSummary[]>('/participant-summaries',{},signal); },
  async getSubjectSummary(id,signal) {
    try { return await teamApi.shared<SubjectSummary>('/participant-summaries/'+encodeURIComponent(id),{},signal); }
    catch (error) { if (error instanceof ApiError && error.status === 404) return null; throw error; }
  },
  async listResultPage(id,offset,signal) {
    const page = await teamApi.shared<Omit<ResultPage, 'items'> & {items:(ResultRecord & {preview?:any})[]}>(
      '/participants/'+encodeURIComponent(id)+'/recording-summaries?limit=20&offset='+offset,{},signal);
    if (!Array.isArray(page.items) || page.offset !== offset || page.limit !== 20 || !Number.isInteger(page.total))
      throw Error('목록 응답 형식을 확인하지 못했습니다.');
    return {...page, items:page.items.map(({preview,...row}) => preview ? {
      ...row, gdi:visitGdi(preview).value, cadence:walkingEstimate(preview).cadence,
    } : row)};
  },
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
  async updateCaptureDay(id, rid, day, previousDay) {
    await teamApi.shared(capturePath(id) + '/' + encodeURIComponent(rid) + '/capture-day', {
      method: 'PATCH', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({day, previous_day: previousDay}),
    });
  },
  async deleteResult(id,rid) { await teamApi.shared(capturePath(id)+'/'+encodeURIComponent(rid),{method:'DELETE'}); },
  async analyzeVideo(id,file,progress,signal,capture) {
    const key = pendingKey(id);
    return withPendingLock(key, async (lease) => {
      const check = async () => { await checkLease(lease);
        signal?.throwIfAborted();
        if (pendingKey(id) !== key) throw Error('원래 계정으로 로그인한 뒤 저장 요청을 복구해 주세요.');
      };
      let pending = await readPending(key);
      await check();
      if (!pending) {
        if (!file) throw Error('복구할 요청이 없습니다. 영상을 선택해 주세요.');
        const packet = await extractCapture(file,id,capture?.day ?? '',capture?.mode ?? 'guided',
          signal ?? new AbortController().signal,progress,capture?.height ?? null,capture?.daySource);
        await check();
        const body = JSON.stringify({fileName:file.name,packet});
        if (new TextEncoder().encode(body).byteLength > 8000000) throw Error('분석 데이터가 너무 큽니다. 더 짧은 영상을 선택해 주세요. 서버에는 전송하지 않았습니다.');
        checkPendingBody(body, packet.requestId, id);
        pending = await keepPending({key, id:packet.requestId, subjectId:id, fileName:file.name,
          body, delivery: 'unsent'}, lease);
      }
      await check();
      progress(95,'이전에 저장됐는지 확인하고 같은 요청으로 이어갑니다.');
      try {
        let row: ResultRecord;
        try { row = await getRecord(pending.id,signal); }
        catch (error) {
          if (!isMissing(error)) throw error;
          await check();
          checkPendingBody(pending.body, pending.id, id);
          const knownUnsent = pending.delivery === 'unsent' || pending.delivery === 'rejected';
          // Persist uncertainty BEFORE sending: a crash or lost response must never enable cleanup.
          await markDelivery(key, pending.id, 'unconfirmed', undefined, lease);
          await check();
          let receipt: {id:string;participant_id:string};
          try {
            receipt = await teamApi.shared<{id:string;participant_id:string}>(capturePath(id), {
              method:'POST',headers:{'Content-Type':'application/json'},body:pending.body,
            },signal);
          } catch (error) {
            if (isRejected(error)) {
              await markDelivery(key, pending.id, knownUnsent ? 'rejected' : 'unconfirmed', error.status, lease);
              throw Error(knownUnsent
                ? '서버가 이 저장 요청을 거절했습니다. 영상을 다시 선택해 분석해 주세요. 같은 문제가 반복되면 담당자에게 알려 주세요.'
                : '서버가 요청을 거절했지만 이전 전송의 저장 여부가 불확실해 요청을 보관합니다. 담당자에게 확인해 주세요.');
            }
            throw error;
          }
          if (receipt.id !== pending.id || receipt.participant_id !== id) throw Error('촬영 저장 번호가 일치하지 않습니다.');
          row = await getRecord(receipt.id,signal);
        }
        await check();
        if (row.id !== pending.id || row.subjectId !== id) throw Error('촬영 저장 번호가 일치하지 않습니다.');
        await clearPending(key,pending.id, lease);
        return row;
      } catch (error) {
        if (error instanceof ApiError && error.status === 0) throw Error('저장 여부를 확인하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요. 분석 정보는 보관됩니다.');
        throw Error(errorMessage(error));
      }
    });
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
