from audit import *
import io
def trc(p):
    lines=p.read_text().splitlines();assert lines[2].split()[4]=='m'
    names=[x for x in lines[3].split('\t')[2:] if x];data=np.loadtxt(io.StringIO('\n'.join(lines[6:])))
    coords=data[:,2:].reshape(len(data),len(names),3);assert np.isfinite(coords).all()
    return data[:,1],{n:coords[:,i] for i,n in enumerate(names)},coords

def review(a,t,markers,coords,w,axis):
    tt=np.arange(round((w[1]-w[0])*30)+1)/30+w[0]
    assert tt[0]>=t[0] and tt[-1]<=t[-1]
    values={k:np.column_stack([np.interp(tt,t,v[:,j]) for j in range(3)]) for k,v in markers.items()}
    hh={k:motion(v[:,[0,2]],axis)[0] for k,v in values.items()}
    root=np.column_stack([np.interp(tt,a['time'],a[k]) for k in ['pelvis_tx','pelvis_tz']]);mid=values['midHip'][:,[0,2]]
    discrepancy=(root-root.mean(0))-(mid-mid.mean(0))
    segments=[('RHip','RKnee'),('RKnee','RAnkle'),('LHip','LKnee'),('LKnee','LAnkle')]
    lengths=np.stack([np.linalg.norm(values[x]-values[y],axis=1) for x,y in segments],axis=1)
    native=(t>=w[0])&(t<=w[1]);original=coords[native,:20]
    freeze=max([0]+[(r-l)*float(np.median(np.diff(t))) for l,r in ranges(np.all(np.diff(original,axis=0)==0,axis=(1,2)))])
    vertical=values['midHip'][:,1]
    return {'window':w,'seconds':w[1]-w[0],'midhip_net_horizontal_m':float(np.linalg.norm(mid[-1]-mid[0])),
      'midhip_horizontal_span_m':np.ptp(mid,axis=0).tolist(),'midhip_speed_median':float(np.median(hh['midHip'])),
      'midhip_low_fraction':float(np.mean(hh['midHip'][:-1]<.1)),
      'both_ankles_low_fraction':float(np.mean((hh['LAnkle'][:-1]<.1)&(hh['RAnkle'][:-1]<.1))),
      'ankle_speed_p95':{k:float(np.quantile(hh[k],.95)) for k in ['LAnkle','RAnkle']},
      'original_marker_frozen_seconds':freeze,'leg_length_cv_max':float((lengths.std(0)/lengths.mean(0)).max()),
      'leg_length_range_m':[float(lengths.min()),float(lengths.max())],
      'pelvis_midhip_centered_horizontal_rmse_m':float(np.sqrt(np.mean(discrepancy**2))),
      'midhip_height_range_m':[float(vertical.min()),float(vertical.max())]}

def main():
    rows={m['id']:m for m in json.loads((PREV/'common176/manifest.json').read_text())}
    audits={r['id']:r for r in json.loads((HERE/'audit.json').read_text())};result=[]
    for receipt in json.loads((HERE/'acquisition.json').read_text()):
        if not receipt['name'].endswith('.trc'):continue
        p=HERE/'raw'/receipt['name'];assert sha(p)==receipt['sha256'];id=p.name.split('_')[0];m=rows[id];r=audits[id]
        a=read(TURN/'raw'/m['turn_file']);t,markers,coords=trc(p)
        windows=[('selected_candidate',r['candidate'])]+[(e['kind']+'_event',e['window']) for e in r['events']]
        if id=='sub-397':windows.append(('previously_flagged11to21',[11.,21.]))
        result.append({'id':id,'target':m['target'],'trc_sha256':receipt['sha256'],'marker_names':list(markers),'reviews':[{'kind':name,**review(a,t,markers,coords,w,m['axis'])} for name,w in windows]})
    save('marker-review.json',result)
    print(json.dumps([{**{k:r[k] for k in ['id','target']},'reviews':[x for x in r['reviews'] if x['kind'] in ['selected_candidate','previously_flagged11to21']]} for r in result]),flush=True)
if __name__=='__main__':main()
