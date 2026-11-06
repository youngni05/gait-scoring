"""Create a review inventory; export only explicitly reviewed records to the frozen CNN.

The source video ID is a dependency cluster, not a participant identity.
This preparation does not start video downloads, pose extraction or inference.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from audit_gavd import HERE, DATA, write_csv, write_json

ROOT = HERE.parents[2]
BUNDLE = ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
VIEWS = {'left side','right side','front','back'}
PATHOLOGICAL = {'parkinsons','cerebral palsy','myopathic','antalgic','stroke','prosthetic'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def stats(rows):
    return {label:dict(sequences=len(items), videos=len({r['video_id'] for r in items}))
            for label in ['Normal Gait','Abnormal Gait']
            for items in [[r for r in rows if r['label']==label]]}


def inventory():
    rows=read_csv(HERE/'sequences.csv')
    access={r['video_id']:r for r in read_csv(HERE/'video_access.csv')}
    stream_path=DATA/'stream_probe_results.json'
    streams=json.loads(stream_path.read_text(encoding='utf-8')) if stream_path.exists() else {}
    length=[r for r in rows if r['length_candidate']=='1']
    consistent=[r for r in length if r['label_consistent']=='1' and r['view'] in VIEWS]
    public=[r for r in consistent if access.get(r['video_id'],{}).get('status')=='oembed_available']
    formats=[r for r in public if streams.get(r['video_id'],{}).get('status')=='video_formats_listed']
    fps_inventory=[]
    for label in ['Normal Gait','Abnormal Gait']:
        for video in sorted({r['video_id'] for r in formats if r['label']==label}):
            meta=streams[video]
            fps_inventory.append(dict(label=label,video_id=video,
                                      fps_options=json.dumps(meta['fps_options']),
                                      duration_seconds=meta['duration'],
                                      provenance='stream listing; actual decoded FPS/frame alignment pending'))
    write_csv(HERE/'candidate_video_fps.csv',fps_inventory)
    stages={'all_annotations':stats(rows), 'length_and_stable_metadata':stats(length),
            'consistent_label_known_view':stats(consistent), 'oembed_available':stats(public),
            'video_formats_listed':stats(formats)}
    grouped=[]
    for view in sorted(VIEWS):
        for label,counts in stats([r for r in formats if r['view']==view]).items():
            grouped.append(dict(view=view,label=label,**counts))
    stagedict=dict(stages=stages, format_listed_by_view=grouped,
                  format_listed_sides=stats([r for r in formats if r['view'] in {'left side','right side'}]),
                  format_listed_pathological=stats([r for r in formats if r['label']=='Normal Gait' or r['gait_pattern'] in PATHOLOGICAL]),
                  label_conflicts=stats([r for r in rows if r['label_consistent']=='0']),
                  actual_input_ready=0, note='Source-video counts are not people counts; shared videos can occur in both labels/views.')
    write_json(HERE/'candidate_summary.json',stagedict)
    write_csv(HERE/'candidates_by_view.csv',grouped)
    conflicts=[r for r in rows if r['label_consistent']=='0' or r['metadata_conflicts']]
    write_csv(HERE/'annotation_review.csv',conflicts)
    plan=[]
    for r in formats:
        review=dict(record_id=r['seq'], video_id=r['video_id'], url=r['url'],label=r['label'],
                    gait_pattern=r['gait_pattern'],view=r['view'],
                    annotation_frame_first=r['frame_first'],annotation_frame_last=r['frame_last'],
                    annotation_video_info=r['annotation_video_info'],
                    json_dir=str(DATA/'body25'/r['seq']),selection_csv=str(DATA/'body25'/r['seq']/'selection.csv'),
                    cluster_id='video:'+r['video_id'],identity_basis='source_video_only_not_person',
                    fps='',fps_basis='',annotation_to_decoded_frame_offset='',
                    frame_mapping_reviewed='',target_and_anatomical_sides_reviewed='',
                    walking_and_view_reviewed='',label_reviewed='',include='',review_notes='')
        plan.append(review)
    # Rebuilding inventory must not overwrite a manually reviewed file.
    if plan: write_csv(HERE/'review_template.csv',plan)
    config=dict(status='prepared_not_executed',data_root=str(DATA),bundle=str(BUNDLE),
                model_manifest_sha256=hashlib.sha256((BUNDLE/'models.json').read_bytes()).hexdigest(),
                bundle_manifest_sha256=hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest(),
                folds=[0,1,2,3,4],time_policy='native_frames_exploratory',
                training=False,calibration_refitting=False,score_clipping=False,
                label_policy='Preserve both source fields; inconsistent labels held out pending review.',
                comparison='Within-view normal vs abnormal, exact decoded-FPS strata; pathological-pattern subset separate.',
                unit='Verified person if available, otherwise source-video cluster; never individual windows.',
                planned_statistics=['score distribution','normal-minus-abnormal mean difference',
                                    'AUROC with higher score defined as normal','cluster bootstrap 95% interval'],
                decision_threshold=None,clinical_GDI_accuracy_claim=False,
                outputs=dict(prepared=str(DATA/'prepared'),predictions=str(DATA/'predictions')))
    write_json(HERE/'experiment_config.json',config)
    print(json.dumps(stagedict,ensure_ascii=False,indent=2))


def export(review, out):
    """Do not create runnable inputs until real JSON and all review gates exist."""
    candidates={r['record_id']:r for r in read_csv(HERE/'review_template.csv')}
    records=[];labels=[];seen=set()
    for r in read_csv(review):
        if r['include']!='yes':continue
        key=r['record_id']
        if key in seen:raise ValueError('Duplicate record: '+key)
        seen.add(key)
        if key not in candidates:raise ValueError('Not an audited candidate: '+key)
        source=candidates[key]
        for field in ['video_id','label','gait_pattern','view','annotation_frame_first','annotation_frame_last']:
            if r[field]!=source[field]:raise ValueError('Source metadata changed: '+field+' '+key)
        for gate in ['frame_mapping_reviewed','target_and_anatomical_sides_reviewed',
                     'walking_and_view_reviewed','label_reviewed']:
            if r[gate]!='yes':raise ValueError('Unreviewed '+gate+': '+key)
        int(r['annotation_to_decoded_frame_offset'])
        fps=float(r['fps'])
        if not math.isfinite(fps) or fps<=0 or not r['fps_basis'].strip():
            raise ValueError('Missing decoded FPS provenance: '+key)
        if not r['cluster_id'] or not r['identity_basis']:
            raise ValueError('Missing dependency cluster: '+key)
        folder=Path(r['json_dir']);selection=Path(r['selection_csv'])
        if not folder.is_absolute() or not selection.is_absolute():raise ValueError('Use absolute source paths')
        if not folder.is_dir() or not selection.is_file():raise ValueError('Missing BODY25 or selection CSV: '+key)
        jsons=list(folder.glob('*_keypoints.json'))
        expected=int(r['annotation_frame_last'])-int(r['annotation_frame_first'])+1
        if len(jsons)!=min(1499,expected):raise ValueError('Unexpected JSON frame count: '+key)
        ids=[]
        for path in jsons:
            match=re.search(r'_(\d+)_keypoints\.json$',path.name)
            if not match:raise ValueError('Unnumbered frame JSON: '+key)
            ids.append(int(match.group(1)))
        first=int(r['annotation_frame_first'])
        if sorted(ids)!=list(range(first,first+min(1499,expected))):
            raise ValueError('JSON names must preserve annotation frame numbers: '+key)
        selections=read_csv(selection)
        if len(selections)!=len(jsons) or {s['filename'] for s in selections}!={p.name for p in jsons}:
            raise ValueError('Selection must explicitly cover every frame: '+key)
        records.append(dict(record_id=key,person_id=r['cluster_id'],json_dir=str(folder),
                            selection_csv=str(selection),fps=fps,fps_basis=r['fps_basis'],source_dataset='GAVD'))
        labels.append({k:r[k] for k in ['record_id','video_id','cluster_id','identity_basis','label','gait_pattern','view','fps','fps_basis']})
    if not records:raise ValueError('No reviewed real records; no runnable manifest emitted')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'records.json',dict(schema='body25_records_v2',time_policy='native_frames_exploratory',
               coordinate_convention='OpenPose_BODY25_pixel_xy',records=records))
    write_csv(out/'labels.csv',labels)
    print(json.dumps(dict(records=len(records),clusters=len({r['person_id'] for r in records}),
                         participants_count='not inferred from video IDs')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['inventory','export'])
    p.add_argument('--review');p.add_argument('--out')
    a=p.parse_args()
    if a.command=='inventory':inventory()
    elif not a.review or not a.out:p.error('export requires --review and --out')
    else:export(a.review,a.out)
