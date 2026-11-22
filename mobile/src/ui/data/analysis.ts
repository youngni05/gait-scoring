import { BrowserVideoSource } from '../../pose/browser-video-source';
import { MediaPipePoseExtractor, PoseInferenceError } from '../../pose/mediapipe-extractor';
import { extractVideo } from '../../pose/extract-video';
import type { CaptureDaySource } from './source';

import { teamApi } from './teamApi';
import { ApiError } from '../../api/client';

interface ServerPose {
  width: number; height: number; rotation: number; source_sha256: string; engine_version: string;
  frames: {t: number; lm: number[][] | null}[];
}

async function extractOnServer(file: File, signal: AbortSignal) {
  if (new URL(teamApi.base).protocol !== 'https:' || location.protocol !== 'https:')
    throw Error('서버 분석은 HTTPS로 접속한 앱에서 사용할 수 있습니다.');
  const caps = await teamApi.shared<{transport:string; max_bytes:number; model_sha256:string}>('/video-pose', {}, signal);
  if (caps.transport !== 'tls-stream-v1') throw Error('영상 전송 경로를 확인하지 못했습니다.');
  if (!file.size || file.size > caps.max_bytes) throw Error('64 MiB 이하의 MP4 또는 MOV 영상을 선택해 주세요.');
  try {
    const result = await teamApi.shared<ServerPose>('/video-pose', {
      method:'POST', headers:{'Content-Type':'application/octet-stream'}, body:file.slice(),
    }, signal, 180000);
    return {
      width:result.width, height:result.height, orientation:result.width >= result.height ? 'landscape' : 'portrait',
      rotationAppliedDegrees:result.rotation, frames:result.frames, sourceHash:result.source_sha256, modelHash:caps.model_sha256,
      configuration:{model:'pose_landmarker_full', engine_version:result.engine_version, delegate:'CPU'}, inputObject:'mp.Image',
      profile:'team-server-full-cpu-video/1',
    };
  } catch (error) {
    if (error instanceof ApiError && error.status === 429) throw Error('다른 영상이 서버에서 처리 중입니다. 잠시 후 다시 시작해 주세요.');
    if (error instanceof ApiError && error.status === 422) throw Error('서버에서 영상을 읽지 못했습니다. 휴대폰 분석을 선택하거나 MP4·MOV 영상을 확인해 주세요.');
    throw error;
  }
}

export function postJson(path: string, body: unknown, signal: AbortSignal) {
  return teamApi.shared<any>(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) }, signal);
}

async function sha256(bytes: ArrayBuffer) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

async function extractOnDevice(file: File, signal: AbortSignal, progress: (percent:number,detail?:string)=>void) {
  const session = await new BrowserVideoSource(document.createElement('div')).open(file, signal);
  try {
    const asset = await fetch(teamApi.base + '/assets/pose-full.task', {signal});
    if (!asset.ok)
      throw new Error('관절 추출 모델을 불러오지 못했습니다. 앱 연결을 확인해 주세요.');
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
    return {width,height,orientation,rotationAppliedDegrees,frames:run.frames,configuration,inputObject,sourceHash,modelHash,
      profile:'team-browser-full-'+configuration.delegate.toLowerCase()+'-video/1'};
  } finally {
    await session.dispose();
  }
}

export async function extractCapture(file: File, aliasId: string, day: string, mode: 'guided'|'fixed', signal: AbortSignal, progress: (percent:number,detail?:string)=>void, captureHeight:number|null, daySource: CaptureDaySource = day ? 'user_entered' : 'unknown', serverPose=false) {
    if (serverPose) progress(5,'HTTPS로 전송해 서버에서 관절을 추출하고 있습니다.');
    const {width,height,orientation,rotationAppliedDegrees,frames,configuration,inputObject,sourceHash,modelHash,profile} =
      serverPose ? await extractOnServer(file,signal) : await extractOnDevice(file,signal,progress);
    const recording = crypto.randomUUID();
    const captured = day ? day + 'T00:00:00+09:00' : null;
    progress(90);
    return {
      requestId: recording,
      coords: { schema_version:'team-local-coords/1', recording_id: recording, subject:{id:aliasId,height_cm:captureHeight},
        capture:{mode,recorded_at:captured,width,height,orientation},pose:configuration,
        marks:{stand_start:null,stand_end:null}, frames },
      metadata:{recordedAtSource:daySource,rotationAppliedDegrees},
      bridge_context:{profile, source_sha256:sourceHash,model_sha256:modelHash,
        input_object:inputObject,image_geometry:'upright_unmirrored',clock_basis:'video_pts',frame_audit_complete:true,fresh_tracker:true}
    };
}
