"""Proposed, opt-in adapter for two official-OpenPose confidence upper-bound failures.

Keep frozen bundle, raw JSON, model features and all other validation unchanged.
Run only after approval of this input-contract exception, with --approved.
"""
import argparse,hashlib,inspect,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parents[2]/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
ALLOWED={'cljxjv25j001a3n6lw4iv9fwe','cll7ob018002y3o6lngai3iq4'}

def main(args):
    if not args.approved:raise SystemExit('Input-contract exception requires approval; no output written.')
    sys.path.insert(0,str(BUNDLE))
    import prepare
    original=prepare.read_record
    source=inspect.getsource(original)
    before='if (a[2::3] < 0).any() or (a[2::3] > 1).any():'
    assert source.count(before)==1
    adjusted=source.replace(before,'if (a[2::3] < 0).any():')
    scope=dict(original.__globals__);exec(compile(adjusted,'explicit_official_confidence_exception','exec'),scope)
    adapted=scope['read_record'];used=[]
    def reader(row,base):
        if row['record_id'] not in ALLOWED:return original(row,base)
        try:return original(row,base)
        except ValueError as error:
            if str(error)!='Confidence must be in [0,1]':raise
        result=adapted(row,base);used.append(row['record_id']);return result
    prepare.read_record=reader
    result=prepare.prepare(args.manifest,args.out)
    audit={'exception':'Only upper confidence bound waived for two preregistered records; all values retained unchanged',
           'allowed_records':sorted(ALLOWED),'used_records':used,'original_read_record_sha256':hashlib.sha256(source.encode()).hexdigest(),
           'adapter_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'official_reference_evidence':str(HERE/'confidence_reference_check.json'),
           'original_bundle_modified':False,'confidence_clipped':False,'coordinate_modified':False,
           'missingness_gate_modified':False,'training_performed':False,'result':result}
    prepare.save_json(Path(args.out)/'confidence_exception.json',audit);print(json.dumps(audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--approved',action='store_true');main(p.parse_args())
