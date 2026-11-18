import { useEffect, useMemo, useState } from 'react';
import type { PoseFrame } from '../../contracts/coords-1/types';
import { FrontalStandDetector, STAND_CONFIG } from './pose/stand-detector';
export function StandTrial({frames,duration,aspectRatio,onSeek}:{frames:readonly PoseFrame[];duration:number;aspectRatio:number;onSeek:(t:number)=>void}){
 const report=useMemo(()=>new FrontalStandDetector(aspectRatio).analyze(frames,duration,new AbortController().signal),[frames,duration,aspectRatio]);
 const [kind,setKind]=useState('unmarked'),[start,setStart]=useState(''),[end,setEnd]=useState('');
 useEffect(()=>{setKind('unmarked');setStart('');setEnd('');},[frames]);
 const d=report.detection;
 let comparison='사용자가 표시한 정답 구간이 없어 정확도 판정은 미검증입니다.';
 if(kind==='none')comparison=d.status==='detected'?'수동 표시와 불일치: 오검출 후보입니다.':'수동 표시와 일치: 양쪽 모두 미검출입니다.';
 if(kind==='range'){
  const a=Number(start),b=Number(end);
  comparison=!start||!end||!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<=a||b>duration?'0 ≤ 시작 < 끝 ≤ 영상 길이로 입력하세요.':d.status==='not_detected'?'수동 표시와 불일치: 자동 검출 누락 후보입니다.':`자동 − 수동: 시작 ${(d.marks.stand_start-a).toFixed(3)}초 / 끝 ${(d.marks.stand_end-b).toFixed(3)}초. 허용 오차 기준은 미확정입니다.`;
 }
 return <section style={{marginTop:24}}><h2>서 있기 구간 · 실험 규칙</h2>
 <small>{STAND_CONFIG.version} · 정면 영상용 초기 후보. 원본 관절값은 변경하지 않습니다.</small>
 <p role="status" className="status">{d.status==='detected'?`서 있기 후보 ${d.marks.stand_start.toFixed(6)} ~ ${d.marks.stand_end.toFixed(6)}초 (${(d.marks.stand_end-d.marks.stand_start).toFixed(3)}초). 수동 확인이 필요합니다.`:`미검출 · 시작 null / 끝 null. ${d.reason}`}</p>
 {d.status==='detected'&&<div className="actions"><button className="secondary" onClick={()=>onSeek(d.marks.stand_start)}>구간 시작 보기</button><button className="secondary" onClick={()=>onSeek((d.marks.stand_start+d.marks.stand_end)/2)}>구간 중간 보기</button><button className="secondary" onClick={()=>onSeek(d.marks.stand_end)}>구간 끝 보기</button></div>}
 <details><summary>검출 근거와 후보 임계값</summary><p>검사 {report.inspected}프레임 / 자세·가시성 통과 {report.usable}프레임</p><pre style={{overflow:'auto'}}>{JSON.stringify({rejected:report.rejected,config:STAND_CONFIG},null,2)}</pre></details>
 <label>사용자 수동 표시<select value={kind} onChange={e=>setKind(e.target.value)}><option value="unmarked">아직 표시하지 않음</option><option value="range">서 있기 구간 있음</option><option value="none">서 있기 구간 없음</option></select></label>
 {kind==='range'&&<div><label>수동 시작 시각 (초)<input type="number" step="any" value={start} onChange={e=>setStart(e.target.value)}/></label><label>수동 끝 시각 (초)<input type="number" step="any" value={end} onChange={e=>setEnd(e.target.value)}/></label></div>}
 <p role="status">{comparison}</p><small>표시값은 메모리에만 두며 자동 결과를 덮어쓰지 않습니다. 카메라 이동·원근·가림·잘못된 관절로 인한 오검출 가능성이 있습니다. 다인 후보 문제도 별도로 확인해야 합니다.</small>
 </section>;
}
