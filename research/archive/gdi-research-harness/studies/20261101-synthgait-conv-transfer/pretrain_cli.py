"""Pretrain one seed on fixed external walks, without loading GDI outcomes."""
import argparse
from pathlib import Path
import run as r
import stem
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['pilot','main'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    c=r.config();r.verify_manifest();d,tr,va=stem.load_external(r.ROOT,c)
    epochs=3 if a.stage=='pilot' else c['pretrain_max_epochs']
    folder=r.ROOT/'runs/pretrain'/a.stage/('seed%d'%a.seed)
    stem.fit_external(d,tr,va,c,a.seed,epochs,folder,a.device,r)
    for device in ['cuda','cpu'] if a.device=='cuda' else ['cpu']:
        r.setup(7,device);proof=stem.audit_external(folder,c,r,device);r.save(folder/('replay-'+device+'.json'),proof)
        print(proof,flush=True)
if __name__=='__main__':main()
