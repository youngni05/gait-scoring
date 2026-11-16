import type { Landmark, PoseFrame } from '../../../contracts/coords-1/types';
import type { StandDetection, StandDetector } from './ports';
/** Conservative experimental rules, not validated clinical thresholds. Distances use torso length. */
export const STAND_CONFIG = Object.freeze({ version:'stand-frontal-0.1.0', minDuration:1.5, minSamples:8,
 minVisibility:.65, maxGap:.12, minTorso:.025, minKneeAngle:160, maxLeanDegrees:20,
 maxRange:.12, maxScaleChange:.08, maxShoulderChange:.12, maxFrozenSeconds:.5 });
const ids=[11,12,23,24,25,26,27,28,29,30,31,32];
type Point=readonly[number,number];
const distance=(a:Point,b:Point)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const mid=(a:Point,b:Point):Point=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
function angle(a:Point,b:Point,c:Point){const u:Point=[a[0]-b[0],a[1]-b[1]],v:Point=[c[0]-b[0],c[1]-b[1]];const d=Math.hypot(...u)*Math.hypot(...v);return d?Math.acos(Math.max(-1,Math.min(1,(u[0]*v[0]+u[1]*v[1])/d)))*180/Math.PI:0;}
export interface StandReport { detection:StandDetection; inspected:number; usable:number; rejected:{visibility:number;posture:number;motion:number;gap:number;frozen:number} }
export class FrontalStandDetector implements StandDetector {
 constructor(private aspectRatio=1){if(!Number.isFinite(aspectRatio)||aspectRatio<=0)throw Error('유효한 영상 가로/세로 비율이 필요합니다.');}
 async detect(frames:readonly PoseFrame[],duration:number,signal:AbortSignal){return this.analyze(frames,duration,signal).detection;}
 analyze(frames:readonly PoseFrame[],duration:number,signal:AbortSignal):StandReport {
  const c=STAND_CONFIG,rejected={visibility:0,posture:0,motion:0,gap:0,frozen:0};let usable=0;
  if(!Number.isFinite(duration)||duration<=0)throw Error('유효한 영상 길이가 필요합니다.');
  let lastT=-1,previous:readonly Landmark[]|null=null,frozenStart:number|null=null;
  let run:{start:number;end:number;count:number;scale:number;shoulder:number;min:number[];max:number[]}|null=null;
  let best:{start:number;end:number}|null=null;
  const keep=()=>{if(run&&run.end-run.start>=c.minDuration&&run.count>=c.minSamples&&(!best||run.end-run.start>best.end-best.start))best={start:run.start,end:run.end};};
  for(const frame of frames){
   if(signal.aborted)throw new DOMException('취소됨','AbortError');
   if(!Number.isFinite(frame.t)||frame.t<0||frame.t<=lastT||frame.t>duration)throw Error('비정상 프레임 시각');
   const gap=lastT>=0&&frame.t-lastT>c.maxGap;lastT=frame.t;
   if(gap){keep();run=null;previous=null;frozenStart=null;rejected.gap++;}
   const lm=frame.lm;
   if(lm&&(lm.length!==33||lm.some(p=>p.length!==4||p.some(v=>!Number.isFinite(v))||p[3]<0||p[3]>1)))throw Error('비정상 관절 값');
   if(!lm||ids.some(i=>lm[i]![3]<c.minVisibility||lm[i]![0]<0||lm[i]![0]>1||lm[i]![1]<0||lm[i]![1]>1)){keep();run=null;previous=null;frozenStart=null;rejected.visibility++;continue;}
   const same=previous&&ids.every(i=>lm[i]!.every((v,j)=>v===previous![i]![j]));
   if(!same)frozenStart=frame.t;previous=lm;
   if(frozenStart!==null&&frame.t-frozenStart>=c.maxFrozenSeconds){run=null;rejected.frozen++;continue;}
   const point=(i:number):Point=>[lm[i]![0]*this.aspectRatio,lm[i]![1]];
   const shoulder=mid(point(11),point(12)),hip=mid(point(23),point(24)),scale=distance(shoulder,hip),span=distance(point(11),point(12));
   const upright=scale>=c.minTorso&&hip[1]>shoulder[1]&&Math.abs(hip[0]-shoulder[0])/scale<=Math.sin(c.maxLeanDegrees*Math.PI/180)
    &&[23,24].every((i,j)=>point(i)[1]<point(25+j)[1]&&point(25+j)[1]<point(27+j)[1]&&angle(point(i),point(25+j),point(27+j))>=c.minKneeAngle);
   if(!upright){keep();run=null;rejected.posture++;continue;}
   usable++;
   const values=ids.flatMap(i=>[point(i)[0],point(i)[1]]);
   if(run){const lower=values.map((v,i)=>Math.min(v,run!.min[i]!)),upper=values.map((v,i)=>Math.max(v,run!.max[i]!));
    const unstable=values.some((_,i)=>upper[i]!-lower[i]!>c.maxRange*Math.min(scale,run!.scale))||Math.abs(scale/run.scale-1)>c.maxScaleChange||Math.abs(span/run.shoulder-1)>c.maxShoulderChange;
    if(unstable){keep();run=null;rejected.motion++;}else{run.min=lower;run.max=upper;run.end=frame.t;run.count++;}
   }
   if(!run)run={start:frame.t,end:frame.t,count:1,scale,shoulder:Math.max(span,1e-9),min:[...values],max:[...values]};
  }
  keep();
  const chosen=best as {start:number;end:number}|null;
  const detection:StandDetection=chosen?{status:'detected',marks:{stand_start:chosen.start,stand_end:chosen.end},version:c.version}:{status:'not_detected',marks:{stand_start:null,stand_end:null},version:c.version,reason:'가시성·선 자세·연속성·움직임 조건을 만족하는 1.5초 이상 구간이 없습니다. (가림/회전/검출 정지 포함)'};
  return {detection,inspected:frames.length,usable,rejected};
 }
}
