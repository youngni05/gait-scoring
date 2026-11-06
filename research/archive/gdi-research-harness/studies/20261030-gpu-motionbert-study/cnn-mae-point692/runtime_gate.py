"""Project fixed main tasks from measured pilot throughput before local approval."""
import math,time
import run as r
def evaluate(extracted,remote):
 plan=r.load(r.ROOT/'protocol.json');data=r.data();rate={}
 for arm in ['mse','ranksim']:
  curve=r.load(extracted/'runs/pilot'/arm/'pilot/curve.json')
  rate[arm]=max(row['seconds']/row['sample_count'] for row in curve)
 workers={w['rank']:w for w in remote['workers'].values()};rows=[]
 for rank in range(3):
  projected=0.
  for arm,f in plan['tasks']['worker%d'%rank]:
   for j in [k for k in range(5) if k!=f]+[None]:
    tr,_,_=r.split(data,f,j);x=data['x'][tr];p=len(set(data['person_id'][tr]));samples=math.ceil(2*len(x.values)/p)*p
    projected+=rate[arm]*samples*100
  # 25% throughput margin plus 90 minutes for final CPU audits/IO.
  required=projected*1.25+5400
  remaining=24*3600-(remote['server_epoch']-workers[rank]['started_epoch'])-300
  rows.append(dict(rank=rank,projected_main_seconds=projected,required_with_margin_seconds=required,remaining_seconds=remaining,passed=required<remaining))
 return dict(passed=all(x['passed'] for x in rows),workers=rows,pilot_seconds_per_sample=rate,unchanged_max_epochs=100,conditions_shortened=False)
