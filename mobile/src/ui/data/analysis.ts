import { BrowserVideoSource } from '../../pose/browser-video-source';
import { MediaPipePoseExtractor, PoseInferenceError } from '../../pose/mediapipe-extractor';
import { extractVideo } from '../../pose/extract-video';
import type { CaptureDaySource } from './source';

export const openPoseEnabled = import.meta.env.VITE_TEAM_OPENPOSE === 'true';

export async function postJson(path: string, body: unknown, signal: AbortSignal) {
  const response = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body), signal });
  if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('로컬 모델 서버 연결을 확인해 주세요.');
  const data = await response.json();
  if (!response.ok) throw new Error(data.message ?? data.error?.message ?? '모델 계산 실패 (HTTP ' + response.status + ')');
  return data;
}

async function sha256(bytes: ArrayBuffer) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

export async function analyzeFile(file: File, aliasId: string, day: string, mode: 'guided'|'fixed', signal: AbortSignal, progress: (percent:number,detail?:string)=>void, captureHeight:number|null, extractor: 'mediapipe'|'openpose' = 'mediapipe', daySource: CaptureDaySource = day ? 'user_entered' : 'unknown') {
  if (extractor === 'openpose') {
    if (!openPoseEnabled) throw new Error('이 앱은 MediaPipe Full로 관절을 추출합니다.');
    if (file.size > 128 * 1024 * 1024) throw new Error('OpenPose는 128MB 이하 MP4를 지원합니다.');
    const health = await fetch('/health', {signal, cache:'no-store'});
    if (!health.ok || !health.headers.get('content-type')?.includes('application/json') || (await health.json()).scheduling !== 'body25-slurm/1')
      throw new Error('학교 GPU에 연결된 분석 서버를 시작해 주세요.');
    const id = crypto.randomUUID();
    const hash = await sha256(await file.arrayBuffer());
    const statusUrl='/v1/team/openpose-status/'+id;
    progress(2,'학교 GPU 분석을 위해 MP4를 분석 서버에 전달하고 있습니다.');
    const cancel = () => { void fetch(statusUrl,{method:'DELETE',keepalive:true}).catch(()=>{}); };
    signal.addEventListener('abort',cancel,{once:true});
    let stopped=false;
    let timer:ReturnType<typeof setTimeout> | undefined;
    const poll=async()=>{
      try {
        const response=await fetch(statusUrl,{signal,cache:'no-store'});
        if(response.ok && !stopped){
          const state=await response.json();
          if(stopped)return;
          if(state.phase==='queued')progress(2,'앞선 OpenPose 분석 '+state.ahead+'건이 끝나면 자동으로 시작합니다.');
          if(state.phase==='slurm_wait')progress(4,'학교 GPU 배정을 기다리고 있습니다.');
          if(state.phase==='extracting')progress(state.total?5+Math.round(80*state.frames/state.total):5,
            'OpenPose 관절 추출 중 · '+state.frames+(state.total?' / '+state.total:'')+'프레임');
          if(state.phase==='scoring')progress(90,'추출한 좌표로 보행 구간과 모델값을 계산하고 있습니다.');
        }
      } catch { /* The original POST owns success and error reporting. */ }
      if(!stopped && !signal.aborted)timer=setTimeout(poll,1500);
    };
    timer=setTimeout(poll,700);
    try {
    const response = await fetch('/v1/team/openpose-video',{method:'POST',signal,body:file,
      headers:{'Content-Type':'video/mp4','X-Recording':JSON.stringify({id,subject:aliasId,day,daySource,height:captureHeight})}});
    if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('이 PC의 OpenPose 분석 서버를 확인해 주세요.');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message ?? 'OpenPose 분석에 실패했습니다.');
    const captured = day ? day+'T00:00:00+09:00' : null;
    if (data.schema!=='team-local-analysis/1' || data.analysis?.provenance?.extractor?.name!=='OpenPose' || data.analysis.provenance.extractor.delegate!=='GPU' ||
        data.source_sha256!==hash || data.record?.recording_id!==id || data.record.subject_id!==aliasId ||
        data.record.captured_at!==captured) throw new Error('선택한 영상과 OpenPose 결과의 출처가 일치하지 않습니다.');
    return {...data,capture_mode:mode,id,subject_id:aliasId,captured_at:captured,saved_at:null,fileName:file.name};
    } finally {stopped=true;clearTimeout(timer);signal.removeEventListener('abort',cancel);}
  }
  const session = await new BrowserVideoSource(document.createElement('div')).open(file, signal);
  try {
    const asset = await fetch('/team-assets/full.task', {signal});
    if (!asset.ok || !asset.headers.get('content-type')?.includes('application/octet-stream'))
      throw new Error('로컬 자세 모델을 읽지 못했습니다. 팀 분석 서버가 실행 중인지 확인해 주세요.');
    const bytes = await asset.arrayBuffer();
    const sourceHash = await sha256(await file.arrayBuffer());
    const modelHash = await sha256(bytes);
    async function extractWith(delegate: 'GPU' | 'CPU') {
      let engine: MediaPipePoseExtractor;
      try { engine = await MediaPipePoseExtractor.create(signal, 1, 'ImageData', new Uint8Array(bytes), delegate); }
      catch (cause) { signal.throwIfAborted(); throw new PoseInferenceError('관절 추출기를 시작하지 못했습니다.', { cause }); }
      try {
        const run = await extractVideo(session, engine, signal, count => {
          if (count > 2000) throw new Error('이 로컬 분석은 2,000프레임 이하 영상을 지원합니다.');
          progress(Math.round(5 + 80 * count / (session.verification?.expected.length || count)), delegate + '로 관절 추출 중 · '+count+'프레임');
        });
        return { run, configuration: engine.configuration, inputObject: engine.inputObject };
      } finally { await engine.dispose(); }
    }
    const {run, configuration, inputObject} = await extractWith('GPU').catch(error => {
      signal.throwIfAborted();
      if (!(error instanceof PoseInferenceError)) throw error;
      progress(3,'GPU 실행이 어려워 CPU로 처음부터 분석합니다.');
      return extractWith('CPU');
    });
    if (!session.verification?.complete || run.frames.length !== session.verification.expected.length) throw new Error('전체 프레임 시각을 확인하지 못했습니다.');
    const { width, height, orientation, rotationAppliedDegrees } = session.info;
    const recording = crypto.randomUUID();
    const captured = day ? day + 'T00:00:00+09:00' : null;
    progress(90);
    const data = await postJson('/v1/team/analyze-coords', {
      requestId: recording,
      coords: { schema_version:'team-local-coords/1', recording_id: recording, subject:{id:aliasId,height_cm:captureHeight},
        capture:{mode,recorded_at:captured,width,height,orientation},pose:configuration,
        marks:{stand_start:null,stand_end:null}, frames:run.frames },
      metadata:{recordedAtSource:daySource,rotationAppliedDegrees},
      bridge_context:{profile:'team-browser-full-'+configuration.delegate.toLowerCase()+'-video/1', source_sha256:sourceHash,model_sha256:modelHash,
        input_object:inputObject,image_geometry:'upright_unmirrored',clock_basis:'video_pts',frame_audit_complete:true,fresh_tracker:true}
    }, signal);
    if (data.schema !== 'team-local-analysis/1' || data.analysis?.schema !== 'ai-analysis/2' ||
        data.source_sha256 !== sourceHash || data.record?.recording_id !== recording || data.record.subject_id !== aliasId ||
        data.record.captured_at !== captured || data.analysis.provenance?.extractor?.delegate !== configuration.delegate) throw new Error('선택한 영상과 분석 결과의 출처가 일치하지 않습니다.');
    return {...data,capture_mode:mode,id:recording, subject_id:aliasId, captured_at:captured, saved_at:null, fileName:file.name};
  } finally {
    await session.dispose();
  }
}
