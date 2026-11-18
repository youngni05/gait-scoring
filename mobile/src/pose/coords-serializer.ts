import type { Coords1, Landmark } from '../../../contracts/coords-1/types';
import { validateCoords, type ValidationResult } from '../../../contracts/coords-1/validate';
import type { CoordsSerializer, SerializationInput } from './ports';
import { checkProcessingInput } from './processing-input';
function rounded(value:Coords1):Coords1{return {...value,frames:value.frames.map(f=>({t:f.t,lm:f.lm===null?null:f.lm.map(p=>p.map(v=>Number(v.toFixed(5)))as unknown as Landmark)}))};}
export class JsonCoordsSerializer implements CoordsSerializer {
 create(s:SerializationInput):ValidationResult {
  const errors=checkProcessingInput(s.input);if(errors.length)return{ok:false,issues:errors.map(message=>({path:'/input',message}))};
  const raw:Coords1={schema_version:'coords-1',recording_id:s.input.recordingId,subject:{id:s.input.subjectId,height_cm:s.input.heightCm},
   capture:{mode:'guided',recorded_at:s.input.recordedAt,width:s.video.width,height:s.video.height,orientation:s.video.orientation},
   pose:s.pose,marks:s.marks,frames:s.frames};
  // Validate before JSON.stringify/toFixed: invalid values must never become null/zero silently.
  const checked=validateCoords(raw,{durationSeconds:s.video.durationSeconds,expectedSubjectId:s.input.subjectId});
  if(!checked.ok)return checked;
  const copy=JSON.parse(JSON.stringify(rounded(checked.value))) as Coords1;
  return validateCoords(copy,{durationSeconds:s.video.durationSeconds,expectedSubjectId:s.input.subjectId});
 }
 stringify(value:Coords1):string {
  const checked=validateCoords(value);if(!checked.ok)throw Error(checked.issues.map(i=>`${i.path}: ${i.message}`).join('; '));
  const result=rounded(checked.value);const final=validateCoords(result);if(!final.ok)throw Error('직렬화 후 계약 검증 실패');
  return JSON.stringify(result);
 }
}
