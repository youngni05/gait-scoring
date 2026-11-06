"""Score-blind camera-condition pairing; no matching based on predictions."""
import argparse,json,math
from audit_gavd import HERE,write_csv,write_json
from prepare_comparison import read_csv
def main(args):
    rows=read_csv(HERE/('conditions_shift%d.csv'%args.shift));normal=[r for r in rows if r['label']=='Normal Gait'];abnormal=[r for r in rows if r['label']=='Abnormal Gait']
    pairs=[];used=set();audit=[]
    for n in sorted(normal,key=lambda r:(r['view'],float(r['fps']),r['record_id'])):
        options=[]
        for a in abnormal:
            if (a['cluster_id'],a['view']) in used:continue
            if a['view']!=n['view'] or abs(float(a['fps'])-float(n['fps']))>=.001:continue
            if a['environment']!=n['environment'] or a['camera']!=n['camera']:continue
            ratio=float(a['height'])/float(n['height']);body=float(a['bbox_height_pixels'])/float(n['bbox_height_pixels'])
            if not (.5<=ratio<=2 and .5<=body<=2):continue
            distance=abs(math.log(ratio))+abs(math.log(body))
            options.append((distance,a['record_id'],a))
        options.sort(key=lambda t:t[:2])
        if not options:audit.append(dict(normal_record=n['record_id'],reason='no remaining abnormal source matching view/native-FPS/environment/camera and 0.5..2x image/body height'));continue
        distance,_,a=options[0];used.add((a['cluster_id'],a['view']))
        pairs.append(dict(normal_record=n['record_id'],abnormal_record=a['record_id'],normal_cluster=n['cluster_id'],abnormal_cluster=a['cluster_id'],view=n['view'],normal_fps=n['fps'],abnormal_fps=a['fps'],environment=n['environment'],camera=n['camera'],log_size_distance=distance))
    write_csv(HERE/('matched_pairs_shift%d.csv'%args.shift),pairs)
    write_json(HERE/('matching_audit_shift%d.json'%args.shift),dict(pairs=len(pairs),unmatched_normal=audit,rule='Before scores: same view, |FPS difference| <0.001 (29.97003 and29.97000 grouping only, no frame conversion), same reviewed environment/camera category, image-height and bbox-height ratio0.5..2. Greedy nearest log size, deterministic IDs; abnormal cluster not reused in same view.',not_matched=['age','sex','body mass','walking speed','true disease/healthy status'],interpretation='closer acquisition conditions, not a matched clinical case-control study'))
    print(json.dumps(dict(pairs=len(pairs),unmatched=len(audit))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shift',type=int,default=0);main(p.parse_args())
