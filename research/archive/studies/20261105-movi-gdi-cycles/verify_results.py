"""Rebuild delivered counts, links, supplementary scores and file fingerprints."""
import json,hashlib,csv
from pathlib import Path
import numpy as np
from analyze import HERE,AUDIT,REF,score
cycles=json.loads((HERE/'cycles.json').read_text());subjects=json.loads((HERE/'subjects.json').read_text());diag=json.loads((HERE/'cycle-diagnostics.json').read_text());stats=json.loads((HERE/'diagnostics.json').read_text());math=json.loads((HERE/'math-verification.json').read_text());summary=json.loads((HERE/'summary.json').read_text())
data=np.load(HERE/'angle-curves.npz');kin=np.load(HERE/'foot-definition-sensitivity.npz')['ankle_toe_curves'];curves=data['curves']
selected=json.loads((AUDIT/'movi-counts.json').read_text())['PG1']['60']['ids'];assert sorted(s['id'] for s in subjects)==selected
assert len(subjects)==28 and len(cycles)==52 and curves.shape==(52,51,9) and np.isfinite(curves).all()
assert summary['protocol_sha256']==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
unflagged=[];foot_errors=[]
for d in diag:
    i=d['curve_index'];c=cycles[i];assert (c['id'],c['side'])==(d['id'],d['side'])
    flag=bool(np.max(np.abs(curves[i,:,2]))>20 or np.ptp(curves[i,:,2])>20);assert flag==d['body_orientation_flag']
    if not flag:unflagged.append(d)
    foot_errors.append(abs(score(kin[i])-d['ankle_toe_definition_gdi']))
    assert abs(score(curves[i])-c['exploratory_gdi'])<1e-10
    assert np.array_equal(np.delete(kin[i],7,axis=1),np.delete(curves[i],7,axis=1))
assert len(unflagged)==30 and len({d['id'] for d in unflagged})==22
assert stats['orientation_unflagged']['people']==22 and stats['orientation_unflagged']['cycles']==30
assert max(foot_errors)<1e-10 and math['numerical_checks_passed'] and math['synthetic_known_pose_cases']==20
assert not math['clinical_gdi_equivalence_passed'] and math['training_labels_accepted']==0
assert len(list(csv.DictReader((HERE/'cycles.csv').open(encoding='utf-8-sig'))))==52
assert len(list(csv.DictReader((HERE/'cycle-diagnostics.csv').open(encoding='utf-8-sig'))))==52
expected=[HERE/'qa'/f'contacts-{n}.jpg' for n in range(1,8)]+[HERE/'nine-angle-curves.png',HERE/'nine-angle-curves.pdf',HERE/'cycle-example.jpg',HERE/'README.md']
assert all(p.exists() and p.stat().st_size>0 for p in expected)
report={'audit_checks_passed':True,'people_checked':28,'kinematic_cycle_candidates':52,'candidate_cycles_with_nine_finite_angle_curves':52,'orientation_unflagged_people':22,'orientation_unflagged_cycles':30,'original_reference_calculation_reproduced':True,'synthetic_known_pose_cases':20,'score_replay_max_error':math['independent_scalar_score_replay_max_error'],'foot_sensitivity_replay_max_error':max(foot_errors),'source_event_contact_sheets_reviewed_by_assistant':7,'event_frame_positions_reviewed':156,'visual_review_scope':'Cycle correspondence only, not precise forceplate contact timing or all-frame occlusion validation','forceplate_contact_validation_passed':False,'clinical_angle_validation_passed':False,'stanford_label_equivalence_passed':False,'training_labels_accepted':0}
(HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'}
(HERE/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(report))
