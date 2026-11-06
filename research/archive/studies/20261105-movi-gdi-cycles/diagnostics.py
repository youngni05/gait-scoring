import io,tarfile,json,csv,sys
from pathlib import Path
import numpy as np
from analyze import HERE,AUDIT,ROOT,REF,filter_xyz,frame_from_yz,frame_from_xy,relative,euler,resample,score,unit,dump,source_frame
from scipy.io import loadmat

def main():
    cycles=json.loads((HERE/'cycles.json').read_text());subjects=json.loads((HERE/'subjects.json').read_text());original=np.load(HERE/'angle-curves.npz')['curves'];candidate={s['id']:s for s in subjects};old={s['id']:s for s in json.loads((AUDIT/'movi-audit.json').read_text())['subjects']};kinetic=original.copy()
    for name in ['F_Subjects_1_45.tar','F_Subjects_46_90.tar']:
        with tarfile.open(ROOT/name) as t:
            for member in t:
                if not member.name.endswith('.mat'):continue
                sid=int(Path(member.name).stem.split('_')[-1])
                if sid not in candidate:continue
                m=loadmat(io.BytesIO(t.extractfile(member).read()),simplify_cells=True)[f'Subject_{sid}_F']['move'];names=list(m['markerName']);r=old[sid];offset=(r['walking_start']-1)*4;end=(r['walking_end']-1)*4
                raw=m['markerLocation'][offset:end+1].copy();raw[m['markerGaps'][offset:end+1].astype(bool)]=np.nan;xyz=filter_xyz(raw);p=lambda n:xyz[:,names.index(n)]
                for side in ['L','R']:
                    s=1 if side=='L' else -1;thigh=frame_from_yz(s*(p(side+'KNE')-p(side+'KNEIN')),p(side+'HIP')-p(side+'KJC'));shank=frame_from_yz(thigh[:,:,1],p(side+'KJC')-p(side+'AJC'));foot=frame_from_xy(p(side+'TOE')-p(side+'AJC'),shank[:,:,1]);ankle=-euler(relative(shank,foot),'YZX')[:,0]
                    for c in cycles:
                        if c['id']!=sid or c['side']!=side or not c['accepted']:continue
                        a=c['start_mocap_one_based']-1-offset;b=c['end_mocap_one_based']-1-offset;kinetic[c['curve_index'],:,7]=np.interp(np.linspace(a,b,51),np.arange(len(ankle)),ankle)
    results=[]
    for c in cycles:
        if not c['accepted']:continue
        i=c['curve_index'];yaw=original[i,:,2];flag=bool(np.max(np.abs(yaw))>20 or np.ptp(yaw)>20)
        results.append({'id':c['id'],'side':c['side'],'curve_index':i,'body_orientation_flag':flag,'pelvis_yaw_max_abs_deg':float(np.max(np.abs(yaw))),'pelvis_yaw_range_deg':float(np.ptp(yaw)),'exploratory_gdi':c['exploratory_gdi'],'ankle_toe_definition_gdi':score(kinetic[i]),'gdi_difference':score(kinetic[i])-c['exploratory_gdi'],'ankle_definition_mean_difference_deg':float(np.mean(kinetic[i,:,7]-original[i,:,7])),'start_video':c['start_video_nearest'],'end_video':c['end_video_nearest']})
    straight=[r for r in results if not r['body_orientation_flag']];ids=sorted({r['id'] for r in straight});shift=[abs(v-c['exploratory_gdi']) for c in cycles for v in c.get('phase_shift_gdi',{}).values()]
    def group_stats(rs):
        personmeans=[np.mean([r['exploratory_gdi'] for r in rs if r['id']==i]) for i in sorted({r['id'] for r in rs})];diff=np.array([r['gdi_difference'] for r in rs]);return {'people':len(personmeans),'cycles':len(rs),'person_equal_mean_gdi':float(np.mean(personmeans)),'cycle_equal_mean_gdi':float(np.mean([r['exploratory_gdi'] for r in rs])),'cycle_gdi_range':[float(min(r['exploratory_gdi'] for r in rs)),float(max(r['exploratory_gdi'] for r in rs))],'ankle_definition_gdi_absolute_difference_median':float(np.median(abs(diff))),'ankle_definition_gdi_absolute_difference_max':float(np.max(abs(diff))),'ankle_definition_gdi_difference_mean':float(np.mean(diff))}
    stats={'all':group_stats(results),'orientation_unflagged':group_stats(straight),'orientation_unflagged_ids':ids,'orientation_flagged_only_ids':sorted(set(candidate)-set(ids)),'orientation_unflagged_bilateral_people':sum(any(r['id']==i and r['side']=='L' for r in straight) and any(r['id']==i and r['side']=='R' for r in straight) for i in ids),'phase_shift_tests':len(shift),'phase_shift_abs_gdi_difference_median':float(np.median(shift)),'phase_shift_abs_gdi_difference_max':float(max(shift)),'angles':[]}
    for j,n in enumerate(REF['angle_names']):
        y=original[[r['curve_index'] for r in straight],:,j];ref=REF['control_angle_curves'][:,:,j];stats['angles'].append({'name':str(n),'movi_mean_deg':float(y.mean()),'reference_mean_deg':float(ref.mean()),'rmse_to_reference_mean_curve_deg':float(np.sqrt(np.mean((y-ref.mean(0))**2))),'interpretation':'Reference differences are not measured angle errors'})
    dump('diagnostics.json',stats);dump('cycle-diagnostics.json',results);np.savez_compressed(HERE/'foot-definition-sensitivity.npz',ankle_toe_curves=kinetic)
    with (HERE/'cycle-diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
