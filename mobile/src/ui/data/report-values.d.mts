export function prepareKneeSeries(segment:unknown, times:unknown):{available:boolean;x:number[];source_indices:number[];left:(number|null)[];right:(number|null)[]};
export function cardMetric(segment:unknown,report:unknown,kind:string):{value:number|null;reason?:string;sentence?:string};
export function directionLabel(direction:string):string;
export function visitKneeSummary(visit:unknown):{
  side:'left'|'right'; value:number; full:boolean; isolatedGaps:boolean; frames:number; start:number; stop:number;
  duration:number; segmentId:string; direction:string
}[];
