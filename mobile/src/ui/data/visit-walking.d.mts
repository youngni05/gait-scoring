interface StepTime { value:number|null; count:number }
export function walkingEstimate(visit:unknown,direction?:string):{
  cadence:number|null; left:StepTime; right:StepTime; count:number;
  intervals:{segmentId:string;direction:string;start_ms:number;stop_ms:number;side:string;seconds:number}[]
};
export function priorVisits<T>(current:unknown,priors:T[]):T[];
export function conditionNote(a:unknown,b:unknown):string[];
export function directionHistory(current:unknown,priors:unknown[]):{sentence:string;groups:{
  direction:string; currentSegments:number; previousSegments:number; previousDate:string; previousId:string;
  notes:string[]; rows:{key:string;label:string;unit:string;current:number;previous:number;currentCount:number;previousCount:number}[]
}[]};
