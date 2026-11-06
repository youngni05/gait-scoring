"""Read Slurm capacity. Never reserve, cancel, or modify other jobs."""
import argparse,datetime,json,re,subprocess


def read_capacity(text):
    rows=[]
    for part in re.split(r'(?=NodeName=)',text):
        if not part.startswith('NodeName='):continue
        def field(name):
            m=re.search(r'\b'+re.escape(name)+r'=([^\s]*)',part)
            return m.group(1) if m else ''
        state=field('State')
        def count(value):
            m=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',value)
            return int(m.group(1)) if m else 0
        total=count(field('CfgTRES'));used=count(field('AllocTRES'))
        eligible=not any(s in state for s in ('DOWN','DRAIN','FAIL','UNKNOWN','MAINT','NO_RESPOND'))
        rows.append(dict(node=field('NodeName'),state=state,total=total,allocated=used,free=max(0,total-used) if eligible else 0))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--min-free',type=int,required=True);a=p.parse_args()
    raw=subprocess.check_output(['scontrol','show','nodes','DIS01,DIS02,DIS03,DIS04,DIS05,DIS06'],text=True)
    rows=read_capacity(raw)
    if len(rows)!=6:raise RuntimeError('Incomplete node capacity response')
    result=dict(time=datetime.datetime.now().isoformat(),nodes=rows,total_free=sum(r['free'] for r in rows),minimum_required=a.min_free)
    print(json.dumps(result),flush=True)
    if result['total_free']<a.min_free:raise SystemExit('Not enough spare GPUs; no training started')


if __name__=='__main__':main()
