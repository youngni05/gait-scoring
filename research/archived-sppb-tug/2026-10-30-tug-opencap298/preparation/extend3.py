from prepare import *
primary=pd.read_csv(HERE/'3s/manifest.csv');rows=primary.to_dict('records');seen=set(primary.id)
records=json.loads((HERE/'record-audit.json').read_text())
for r in sorted(records,key=lambda r:(r['id'],r['visit'],r['session'])):
    if r['id'] in seen or r['target'] is None or r['target']<=0 or r['active_max_seconds']<3:continue
    a=read(HERE/'raw'/r['file']);start=next(s for s,e,dur in active(a) if dur>=3)
    rows.append(dict(r,start_frame=start,start_time=float(a['time'][start]),seconds=3,source_sha256=sha(HERE/'raw'/r['file'])));seen.add(r['id'])
out=HERE/'all3s';out.mkdir(exist_ok=True)
base=dict(np.load(HERE/'3s/input.npz'));extra=rows[len(primary):];xs=[window(read(HERE/'raw'/r['file']),int(r['start_frame']),3) for r in extra]
folds=np.zeros(len(extra),int)
for f,(_,ix) in enumerate(KFold(5,shuffle=True,random_state=20261030).split(extra)):folds[ix]=f
np.savez_compressed(out/'input.npz',x=np.concatenate([base['x'],np.stack(xs)]),y=np.array([r['target'] for r in rows]),ids=np.array([r['id'] for r in rows]),folds=np.r_[base['folds'],folds],frames=np.full(len(rows),91))
pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
protocol=json.loads((HERE/'3s/protocol.json').read_text());protocol.update(people=len(rows),cohort='secondary expanded3s cohort; primary196samevisit/start/fold retained; additionalpeople first eligible3s visit/session; same person splitting principle',comparison_limit='different cohort from primary1/3/6s; do not interpret metric difference as duration effect')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
(out/'input-audit.json').write_text(json.dumps({'input_sha256':sha(out/'input.npz'),'protocol_sha256':sha(out/'protocol.json')},indent=2))
for n in ['common.py','cpu.py','gpu.py']:shutil.copy2(HERE/'3s'/n,out/n)
print('EXPANDED3',len(rows),'ADDITIONAL',len(extra))
