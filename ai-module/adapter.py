"""Transport-only bridge: audited browser coordinates to unchanged analyze/2."""
import hashlib,json,math,re
from datetime import datetime
from pathlib import Path
from statistics import median
from uuid import UUID
from schema import fields,finite,require,text,strict_value,LIMITS
from prepare_assets import POSE_MODEL_SHA256

SAMPLING='native-video-pts-no-resampling/1'
SETTINGS=dict(runningMode='VIDEO',numPoses=1,minPoseDetectionConfidence=.5,minPosePresenceConfidence=.5,
              minTrackingConfidence=.5,outputSegmentationMasks=False)
HASH=re.compile(r'[0-9a-f]{64}')
DATE=re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def checked_marks(m,path,last):
    fields(m,('stand_start','stand_end'),path)
    a,b=m['stand_start'],m['stand_end']
    require(a is None and b is None or finite(a) and finite(b) and 0<=a<b<=last,'Standing marks must be both null or an ordered interval in observed time coverage',path)

def to_request(body):
    strict_value(body)
    fields(body,('requestId','coords','metadata','bridge_context'),'$')
    text(body['requestId'],'$.requestId')
    c=body['coords'];fields(c,('schema_version','recording_id','subject','capture','pose','marks','frames'),'$.coords')
    require(c['schema_version'] in ('coords-1','team-local-coords/1'),'Unsupported local coordinate schema','$.coords.schema_version')
    local=c['schema_version']=='team-local-coords/1'
    text(c['recording_id'],'$.coords.recording_id')
    require(re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',c['recording_id']) is not None,'Expected recording UUID','$.coords.recording_id')
    UUID(c['recording_id'])
    subject=c['subject'];fields(subject,('id','height_cm'),'$.coords.subject');text(subject['id'],'$.coords.subject.id')
    require(bool(subject['id'].strip()),'Expected nonblank pseudonym','$.coords.subject.id')
    height=subject['height_cm'];require(local and height is None or finite(height) and 100<=height<=220,'Invalid height, local unknown must be null','$.coords.subject.height_cm')
    capture=c['capture'];fields(capture,('mode','recorded_at','width','height','orientation'),'$.coords.capture',('device',))
    require(capture['mode'] in ('guided','fixed'),'Unsupported capture mode','$.coords.capture.mode')
    for key in ('width','height'):
        require(type(capture[key]) is int and 0<capture[key]<=LIMITS['max_image_dimension'],'Invalid upright pixel dimensions','$.coords.capture.'+key)
    require(capture['orientation'] in ('landscape','portrait') and
            (capture['width']==capture['height'] or (capture['width']>capture['height'])==(capture['orientation']=='landscape')),
            'Orientation must match rotated dimensions','$.coords.capture.orientation')
    if 'device' in capture:text(capture['device'],'$.coords.capture.device')
    date=capture['recorded_at']
    if not (local and date is None):
        require(isinstance(date,str) and DATE.fullmatch(date) is not None,'Expected timezone capture date or local null','$.coords.capture.recorded_at')
        try:stamp=datetime.fromisoformat(date.replace('Z','+00:00'));valid=stamp.tzinfo is not None and stamp.utcoffset() is not None
        except ValueError:valid=False
        require(valid,'Invalid capture date','$.coords.capture.recorded_at')
    pose=c['pose'];fields(pose,('model','engine_version'),'$.coords.pose',('delegate',))
    delegate=pose.get('delegate')
    require(delegate in ('CPU','GPU') and pose['model']=='pose_landmarker_full' and pose['engine_version']=='1.2.3',
            'Expected Full 1.2.3 and actual CPU/GPU delegate','$.coords.pose')
    context=body['bridge_context']
    fields(context,('profile','source_sha256','model_sha256','input_object','image_geometry','clock_basis','frame_audit_complete','fresh_tracker'),'$.bridge_context')
    native=context['profile']=='team-server-full-cpu-video/1'
    frames=c['frames'];require(isinstance(frames,list) and (1 if local else 60)<=len(frames)<=LIMITS['max_frames'],'Invalid frame count','$.coords.frames')
    previous=-1;output=[]
    for i,f in enumerate(frames):
        path=f'$.coords.frames[{i}]';fields(f,('t','lm'),path)
        require(finite(f['t']) and f['t']>=0 and f['t']>previous,'Original PTS must be finite, nonnegative and strictly increasing',path+'.t');previous=f['t']
        lm=f['lm']
        if lm is not None:
            require(isinstance(lm,list) and len(lm)==33,'Expected33 MediaPipe landmarks or null',path+'.lm')
            for j,point in enumerate(lm):
                require(isinstance(point,list) and len(point)==4 and all(finite(v) for v in point) and 0<=point[3]<=1,
                        'Expected finite[x,y,z,visibility] without clipping',path+f'.lm[{j}]')
        output.append(dict(index=i,source_time_ms=f['t']*1000,detector_time_ms=round(f['t']*1000) if native else f['t']*1000,
            landmarks=None if lm is None else [[*point,None] for point in lm],why_unavailable='no_pose' if lm is None else None))
    checked_marks(c['marks'],'$.coords.marks',frames[-1]['t'])
    m=body['metadata'];fields(m,('recordedAtSource','rotationAppliedDegrees'),'$.metadata',('detection','modelAssetVersion'))
    require(m['recordedAtSource'] in ('metadata','user_entered','unknown'),'Invalid date source','$.metadata.recordedAtSource')
    require((date is None)==(m['recordedAtSource']=='unknown'),'Unknown capture date must remain null with source unknown','$.metadata.recordedAtSource')
    require(m['rotationAppliedDegrees'] is None or finite(m['rotationAppliedDegrees']) and m['rotationAppliedDegrees'] in (0,90,180,270,-90,-180,-270),
            'Invalid rotation metadata','$.metadata.rotationAppliedDegrees')
    if 'modelAssetVersion' in m:text(m['modelAssetVersion'],'$.metadata.modelAssetVersion')
    if 'detection' in m:
        d=m['detection'];fields(d,('status','marks','version'),'$.metadata.detection',('reason',))
        require(d['status'] in ('detected','not_detected'),'Invalid stand detector status','$.metadata.detection.status')
        text(d['version'],'$.metadata.detection.version')
        require(d['marks']==c['marks'],'Standing metadata and coords marks differ','$.metadata.detection.marks')
        require((d['status']=='not_detected')==(c['marks']['stand_start'] is None),'Standing status differs from marks','$.metadata.detection')
        if d['status']=='not_detected':text(d.get('reason'),'$.metadata.detection.reason')
    require((native and local and delegate=='CPU') or context['profile']==f'team-browser-full-{delegate.lower()}-video/1','Extractor profile differs from delegate','$.bridge_context.profile')
    require(context['source_sha256']=='unknown' or isinstance(context['source_sha256'],str) and HASH.fullmatch(context['source_sha256']) is not None,
            'Expected original File SHA256 or unknown','$.bridge_context.source_sha256')
    require(context['model_sha256']==POSE_MODEL_SHA256,'Task model differs from checked full model','$.bridge_context.model_sha256')
    require(context['input_object'] in (('mp.Image',) if native else ('ImageData','HTMLVideoElement','unknown')),'Unexpected extractor input','$.bridge_context.input_object')
    require(context['image_geometry']=='upright_unmirrored','Analysis coordinates must be upright and unmirrored','$.bridge_context.image_geometry')
    require(context['clock_basis']=='video_pts','Native PTS required','$.bridge_context.clock_basis')
    require(context['frame_audit_complete'] is True and context['fresh_tracker'] is True,'Complete source frame audit and fresh tracker are required','$.bridge_context')
    source=context['source_sha256'];source_id='sha256:'+source if source!='unknown' else 'recording:'+c['recording_id']
    low=output[0]['source_time_ms'];high=math.nextafter(output[-1]['source_time_ms'],math.inf)
    result=dict(schema='ai-analyze-request/2',request_id=body['requestId'],pose=dict(schema='measurement-pose/1',source_id=source_id,
        image=dict(width=capture['width'],height=capture['height'],orientation='upright_unmirrored',preview_mirrored=None),
        clock=dict(unit='ms',basis='video_pts'),extractor=dict(name='MediaPipe Pose Landmarker',variant='full',
            runtime='mediapipe-python' if native else 'mediapipe-tasks-vision-web',runtime_version=pose['engine_version'],model_sha256=context['model_sha256'],delegate=delegate,settings=dict(SETTINGS)),frames=output),
        provenance=dict(input_object=context['input_object'],tracking=dict(session_id=('team-server:' if native else 'team-browser:')+c['recording_id'],
            start_time_ms=low,input_start_ms=low,input_stop_ms=high,prior_frame_count=0,fresh_tracker=True)))
    return result

def curve_times(frames,analysis):
    clocks={f['index']:f['source_time_ms'] for f in frames};out={}
    for s in analysis['segments']:
        times=[clocks[i] for i in s['observation']['source_indices']]
        for view in ('observed','supported'):
            assert all(len(s['knee']['views'][view]['curve_deg'][side])==len(times) for side in ('left','right'))
        out[s['segment_id']]=times
    return out

def analyze_coords(body):
    request=to_request(body);c=body['coords']
    record=dict(subject_id=c['subject']['id'],recording_id=c['recording_id'],captured_at=c['capture']['recorded_at'],
                height_cm=c['subject']['height_cm'],recorded_at_source=body['metadata']['recordedAtSource'])
    delegate=c['pose']['delegate'].lower()
    profile=body['bridge_context']['profile']
    tracking='team-server-full-cpu-fresh-native-allframes/1' if profile=='team-server-full-cpu-video/1' else f'team-browser-full-{delegate}-fresh-native-allframes/1'
    return finish(request,record,body['bridge_context']['source_sha256'],c,profile,tracking)

def finish(request,record,source,coords,profile,tracking):
    from core import analyze
    from local_observations import build_local_observations, reviewed_recover
    from local_steps import local_step_intervals
    analysis=analyze(request, review_contacts=request['pose']['extractor']['name']=='MediaPipe Pose Landmarker');frames=request['pose']['frames']
    times=[f['source_time_ms'] for f in frames];dt=[b-a for a,b in zip(times,times[1:])]
    sampling=dict(frame_count=len(frames),missing_pose_frames=sum(f['landmarks'] is None for f in frames),
        median_interval_ms=median(dt) if dt else None,minimum_interval_ms=min(dt) if dt else None,
        maximum_interval_ms=max(dt) if dt else None,effective_rate_hz=(len(times)-1)*1000/(times[-1]-times[0]) if dt else None,
        first_source_ms=times[0],last_source_ms=times[-1],resampled=False)
    return dict(schema='team-local-analysis/1',analysis=analysis,curve_time_ms_by_segment=curve_times(frames,analysis),
        local_step_intervals_by_segment=local_step_intervals(request['pose'],analysis),
        local_observations_by_segment=build_local_observations(analysis,frames,reviewed_recover),
        record=record,
        source_sha256=source,coords_sha256=digest(coords),tracking_policy_id=tracking,sampling_policy_id=SAMPLING,
        bridge=dict(schema='team-model-bridge/1',profile=profile,metadata_basis='reported',sampling=sampling,
            input_object=request['provenance']['input_object'],presence='not provided by coords transport; preserved as null',
            date_and_height_used_by_model=False,standing_marks_used_as_gait_segments=False,
            transport_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('adapter.py','team_server.py')}))
