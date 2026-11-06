"""Freeze first comparison cohort before seeing any CNN scores."""
import collections,json,math
from audit_gavd import HERE,DATA,write_json,write_csv
from prepare_comparison import read_csv
def main():
    rows=read_csv(HERE/'review_template.csv');quality={r['record_id']:r for r in read_csv(HERE/'qa/pilot/quality.csv')}
    indexes={r['seq']:r['index'] for r in json.loads((HERE/'qa/normal_video/index.json').read_text())}
    meta=json.loads((DATA/'video_inventory.json').read_text());streams=json.loads((DATA/'stream_probe_results.json').read_text())
    chosen=[];audit=[];used=set()
    for r in rows:
        if r['label']!='Normal Gait':continue
        q=quality[r['record_id']];v=r['video_id'];idx=indexes[r['record_id']];reason=''
        if idx in [27,32,33]:reason='scene_transition_or_unresolved_presentation'
        elif v in ['MN4vnaNwIsA','EHymg4AGMJs']:reason='oblique_view_or_shuttle_protocol_not_matched'
        elif int(q['selected'])!=3 or float(q['max_required_zero_fraction'])>0 or float(q['mean_toe_confidence'])<.1:reason='pilot_required_joint_or_target_missing'
        elif (v,r['view']) in used:reason='first_comparison_one_sequence_per_source_and_view'
        else:chosen.append(r);used.add((v,r['view']))
        audit.append(dict(index=idx,record_id=r['record_id'],video_id=v,view=r['view'],selected_for_full=not bool(reason),reason=reason or 'sampled_visual_and_pose_check_passed_full_check_pending'))
    write_csv(HERE/'normal_selection_audit.csv',audit);write_csv(HERE/'normal_selected.csv',chosen)
    jobs={r['record_id']:r for r in json.loads((HERE/'normal_jobs.json').read_text())['records']}
    full=[]
    for r in chosen:
        job=jobs[r['record_id']].copy()
        # Preserve a trailing neighbour for a predeclared +1-frame alignment sensitivity check.
        job['primary_decoded_last']=job['decoded_last'];job['decoded_last']=min(meta[r['video_id']]['frames']-1,job['decoded_last']+1)
        full.append(job)
    write_json(HERE/'normal_full_jobs.json',dict(records=full,selection='normal_selection_audit.csv',annotation_offset=-1,sensitivity='trailing neighbouring frame included; primary input retains original annotation range'))
    abnormal=[r for r in rows if r['label']=='Abnormal Gait'];selected={};matching=[];usedvideos=set()
    for n in chosen:
        nm=meta[n['video_id']];candidates=[]
        for a in abnormal:
            if a['view']!=n['view']:continue
            fps=streams[a['video_id']]['fps_options']
            if min(abs(float(f)-nm['fps']) for f in fps)>.06:continue
            ai=json.loads(a['annotation_video_info']);ni=json.loads(n['annotation_video_info'])
            distance=abs(math.log(ai['height']/nm['height']))+abs(math.log((ai['width']/ai['height'])/(nm['width']/nm['height'])))
            candidates.append((a['video_id'] in usedvideos,distance,a['video_id'],int(a['annotation_frame_first']),a))
        candidates.sort(key=lambda x:x[:-1]);local=set()
        for _,distance,vid,frame,a in candidates:
            if vid in local:continue
            local.add(vid);selected[a['record_id']]=a;usedvideos.add(vid)
            matching.append(dict(normal_record=n['record_id'],normal_video=n['video_id'],abnormal_record=a['record_id'],abnormal_video=vid,view=n['view'],normal_decoded_fps=nm['fps'],rank=len(local),resolution_log_distance=distance))
            if len(local)==2:break
    write_csv(HERE/'abnormal_screening.csv',list(selected.values()));write_csv(HERE/'matching_screening.csv',matching)
    write_json(HERE/'initial_selection_protocol.json',dict(normal_records=len(chosen),normal_videos=len({r['video_id'] for r in chosen}),abnormal_screening_records=len(selected),abnormal_videos=len({r['video_id'] for r in selected.values()}),normal_frames=sum(r['decoded_last']-r['decoded_first']+1 for r in full),score_blind=True,normal_cap='first eligible annotation sequence per source video and camera-view label',pilot_gate='three unambiguous bbox target matches; all required joints nonzero; mean toe confidence >=0.1; sampled visual review',abnormal_rule='two source-distinct candidates per normal source/view; native FPS metadata within 0.06 then prefer unused sources, closest resolution/aspect ratio, video ID and first annotation frame; actual decoded FPS and visual/pose QC still required',limitations='first comparison cohort, not all usable GAVD; more repeats do not add independent people; labels not verified clinical GDI'))
    print((HERE/'initial_selection_protocol.json').read_text())
if __name__=='__main__':main()
