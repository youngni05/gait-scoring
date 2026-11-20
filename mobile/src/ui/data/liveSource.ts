import { analyzeFile } from './analysis';
import { ageOn, type Subject, type ResultRecord } from './mock';
import { AnalysisFailedError, type GaitDataSource } from './source';
import { visitKneeSummary } from './report-values.mjs';
import { walkingEstimate } from './visit-walking.mjs';

function database() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open('team-template-model-v1',1);
    req.onupgradeneeded = () => {
      for (const key of ['subjects','results']) req.result.createObjectStore(key,{keyPath:'id'});
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function access<T>(store:string, mode:IDBTransactionMode, action:(table:IDBObjectStore)=>IDBRequest<T>) {
  const db=await database();
  try {
    return await new Promise<T>((resolve,reject)=>{
      const tx=db.transaction(store,mode);
      const req=action(tx.objectStore(store));
      tx.oncomplete=()=>resolve(req.result);
      tx.onabort=()=>reject(tx.error ?? req.error);
    });
  } finally { db.close(); }
}
const results=()=>access<ResultRecord[]>('results','readonly',t=>t.getAll());
export const liveSource:GaitDataSource={
  async login(){}, logout(){},
  async listSubjects(){
    const [subjects,records]=await Promise.all([access<Subject[]>('subjects','readonly',t=>t.getAll()),results()]);
    return subjects.map(s=>{
      const mine=records.filter(r=>r.subjectId===s.id).sort((a,b)=>b.measuredAt.localeCompare(a.measuredAt));
      return {...s,latestMeasuredAt:mine[0]?.measuredAt || null,resultCount:mine.length};
    });
  },
  async getSubject(id){return (await access<Subject>('subjects','readonly',t=>t.get(id)))??null;},
  async createSubject(input){const s={...input,id:crypto.randomUUID()}; await access('subjects','readwrite',t=>t.add(s));return s;},
  async updateSubject(id,input){const s={...input,id};await access('subjects','readwrite',t=>t.put(s));return s;},
  async listResults(id){return (await results()).filter(r=>r.subjectId===id).sort((a,b)=>(b.live?.saved_at??'').localeCompare(a.live?.saved_at??''));},
  async getResult(id,rid){const r=await access<ResultRecord>('results','readonly',t=>t.get(rid));return r?.subjectId===id?r:null;},
  async deleteResult(subjectId, resultId) {
    await access('results', 'readwrite', table => {
      const request = table.openCursor(IDBKeyRange.only(resultId));
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor?.value.subjectId === subjectId) cursor.delete();
      };
      return request;
    });
  },
  async analyzeVideo(id,file,progress,signal,capture){
    if(!file)throw new AnalysisFailedError('GA-FILE','실제 MP4 영상을 선택해 주세요.');
    const subject=await this.getSubject(id);
    if(!subject)throw new AnalysisFailedError('GA-404','대상자를 찾지 못했습니다.');
    const day=capture?.day??'';
    const live=await analyzeFile(file,id,day,capture?.mode??'guided',signal??new AbortController().signal,progress,capture?.height??null,capture?.extractor,capture?.daySource);
    if(!live.analysis.segments.length)throw new AnalysisFailedError('GA-SEG','영상에서 직선 보행 구간을 찾지 못했습니다.');
    live.saved_at=new Date().toISOString();
    const r:ResultRecord={id:live.id,subjectId:id,measuredAt:live.captured_at??'',ageAtCapture:day&&subject.birth&&day>=subject.birth?ageOn(subject.birth,new Date(day+'T12:00:00')):null,
      heightAtCapture:capture?.height??null,fileName:file.name,gdi:null,cadence:null,symmetry:null,curves:{},missing:{},live};
    signal?.throwIfAborted();
    await access('results','readwrite',t=>t.add(r));progress(100);return r;
  },
  async downloadResult(id){
    const r=await access<ResultRecord>('results','readonly',t=>t.get(id));
    if(!r)throw Error('저장 기록을 찾지 못했습니다.');
    const report = r.live ? {...r, walking_estimate:walkingEstimate(r.live), knee_observation_summary: {
      unit:'deg', time_unit:'ms', observations:visitKneeSummary(r.live)
    }} : r;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='gait-'+id+'.json';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
};
