"""Run the unchanged teammate v2 preparation with external progress logging.

python run_preparation.py --bundle PATH_TO_UNCHANGED_V2 --manifest records.json --out NEW_DIRECTORY
No training or normal-data inference is performed. Wrapper only logs progress.
"""
import argparse,contextlib,json,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source-zip',type=Path)
    ap.add_argument('--extracted-root',type=Path)
    args=ap.parse_args();sys.path.insert(0,str(args.bundle.resolve()))
    import prepare
    original=prepare.read_record;counts={'attempted':0,'prepared':0,'excluded':0};start=time.monotonic()
    reader=None
    if args.source_zip:
        from zip_input_adapter import ZipInput,install_sha_adapter
        reader=ZipInput(args.source_zip,args.manifest.resolve().parent,args.extracted_root)
        install_sha_adapter(prepare)
    def tracked(row,base):
        counts['attempted']+=1
        try:
            value=original(row,reader if reader else base);counts['prepared']+=1;return value
        except (ValueError,KeyError,TypeError,OSError,IndexError):
            counts['excluded']+=1;raise
        finally:
            if counts['attempted']%25==0:
                print(json.dumps(dict(**counts,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
    prepare.read_record=tracked
    try:result=prepare.prepare(args.manifest,args.out)
    finally:
        if reader:reader.close()
    result['runtime_seconds']=time.monotonic()-start
    result['source_storage']='read_only_zip' if reader else 'extracted_files'
    result['source_archive']=str(args.source_zip.resolve()) if reader else None
    result['extracted_root_mapping']=str(args.extracted_root.resolve()) if args.extracted_root else None
    (args.out/'execution_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',json.dumps(result),flush=True)
    raise SystemExit(0 if result['records_prepared'] else 2)

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
