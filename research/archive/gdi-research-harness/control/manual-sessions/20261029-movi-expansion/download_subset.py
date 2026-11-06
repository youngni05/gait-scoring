"""Two-worker, resumable, checksum-verified acquisition of a fixed MoVi subset."""
import concurrent.futures
import hashlib
import json
import msvcrt
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path('D:/gait-research-data/MoVi')
CONTROL=ROOT/'control'
MANIFEST=CONTROL/'subset-manifest.json'
CURL=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/curl.exe'
guard=threading.RLock()
halt=threading.Event()
state={}
files=[]

def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def save():
    with guard:
        state['updated_unix']=time.time()
        state['verified_files']=sum(v.get('status')=='verified' for v in state['files'].values())
        state['verified_bytes']=sum(v.get('bytes',0) for v in state['files'].values() if v.get('status')=='verified')
        partial_bytes=0
        for p in (ROOT/'raw').glob('*.part'):
            try:partial_bytes+=p.stat().st_size
            except FileNotFoundError:pass  # Other worker completed and renamed it.
        state['partial_bytes']=partial_bytes
        atomic_json(CONTROL/'download-state.json',state)

def update(entry,**values):
    with guard:
        record=state['files'].setdefault(entry['name'],{})
        record.update(values);save()

def hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):md5.update(block);sha.update(block)
    return md5.hexdigest(),sha.hexdigest()

def verify(entry,path):
    if path.stat().st_size!=entry['bytes']:return None
    md5,sha=hashes(path)
    return sha if md5==entry['md5'] else None

def acquire(entry):
    final=ROOT/'raw'/entry['name'];partial=final.with_name(final.name+'.part')
    if halt.is_set():return
    if final.exists():
        sha=verify(entry,final)
        if sha:update(entry,status='verified',bytes=entry['bytes'],sha256=sha,source='existing');return
        update(entry,status='existing_file_mismatch',error='Preserved existing file; not overwritten');halt.set();return
    for attempt in range(1,4):
        if halt.is_set() or (CONTROL/'STOP').exists():return
        current=partial.stat().st_size if partial.exists() else 0
        if current>entry['bytes']:
            update(entry,status='oversize_partial',error='Partial preserved; manual inspection needed');return
        if current==entry['bytes']:
            sha=verify(entry,partial)
            if sha:partial.rename(final);update(entry,status='verified',bytes=entry['bytes'],sha256=sha,source='resumed');return
            update(entry,status='checksum_mismatch',error='Complete partial preserved; not retried');return
        if shutil.disk_usage(ROOT).free<entry['bytes']-current+20_000_000_000:
            update(entry,status='blocked_disk_space');halt.set();return
        update(entry,status='downloading',attempt=attempt,bytes_received=current,expected_bytes=entry['bytes'])
        command=[str(CURL),'-L','--fail','--silent','--show-error','--continue-at','-',
                 '--connect-timeout','20','--max-time','1800','--speed-limit','1024','--speed-time','90',
                 '--max-filesize',str(entry['bytes']),entry['url'],'-o',str(partial),'-w','%{http_code}']
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            time.sleep(5)
            if (CONTROL/'STOP').exists() or halt.is_set():process.terminate()
            update(entry,bytes_received=partial.stat().st_size if partial.exists() else 0)
        output,error=process.communicate();http=output.decode(errors='replace').strip()
        if (CONTROL/'STOP').exists() or halt.is_set():update(entry,status='paused',http_status=http);return
        if http in ['401','403']:
            update(entry,status='blocked_provider_access',http_status=http);halt.set();return
        if partial.exists() and partial.stat().st_size==entry['bytes']:
            sha=verify(entry,partial)
            if sha:
                partial.rename(final);update(entry,status='verified',bytes=entry['bytes'],sha256=sha,http_status=http,source='downloaded')
                print(f"VERIFIED {entry['name']} {entry['bytes']}",flush=True);return
            update(entry,status='checksum_mismatch',http_status=http,error='Preserved corrupt partial; not retried');return
        update(entry,status='retry_pending',curl_exit=process.returncode,http_status=http,error=error.decode(errors='replace')[-700:])
        if http in ['404','410']:update(entry,status='unavailable');return
        if attempt<3:time.sleep(5*attempt)
    update(entry,status='incomplete_after_three_attempts')

def main():
    global state,files
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));files=manifest['files']
    assert sum(e['bytes'] for e in files)==manifest['raw_total_bytes']<=manifest.get('raw_download_budget_bytes',9_000_000_000)
    assert len({e['name'].casefold() for e in files})==len(files)
    for e in files:
        assert re.fullmatch(r'[A-Za-z0-9_. -]+',e['name']) and not e['name'].endswith(('.', ' '))
        assert e['url']==f"https://dataverse.scholarsportal.info/api/access/datafile/{e['id']}"
    # OS lock is released automatically on process exit; lock-file presence alone is not used.
    with (ROOT/'control'/'download.lock').open('a+b') as lock:
        if lock.seek(0,2)==0:lock.write(b'1');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Another downloader owns the lock; no second run started.')
        try:
            saved=CONTROL/'download-state.json'
            state=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {'files':{}}
            state.update(status='running',pid=os.getpid(),started_unix=time.time(),total_files=len(files),total_bytes=manifest['raw_total_bytes'],workers=2)
            save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures={pool.submit(acquire,e):e for e in files}
                for future,entry in futures.items():
                    try:future.result()
                    except Exception as exc:update(entry,status='error',error=f'{type(exc).__name__}: {exc}')
            state['status']='complete' if all(state['files'].get(e['name'],{}).get('status')=='verified' for e in files) else ('paused' if (CONTROL/'STOP').exists() else 'needs_attention')
            state['finished_unix']=time.time();save()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
