"""Acquire and audit an independent step-label candidate before pose or score inference."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,urllib.parse,io,zipfile,struct,time,datetime,os
HERE=Path(__file__).resolve().parent
STORE=Path('D:/gait-research-data/D_P/clemson-step-reference-20261101')
BASE='https://cecas.clemson.edu/~ahoover/pedometer/'
DATA='https://cecas.clemson.edu/tracking/Pedometer/Data.zip'
VIDEOS='https://cecas.clemson.edu/tracking/Pedometer/Videos.zip'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def head(url):
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=25) as r:return dict(url=url,status=r.status,size=int(r.headers['Content-Length']),etag=r.headers.get('ETag'),modified=r.headers.get('Last-Modified'),accept_ranges=r.headers.get('Accept-Ranges'))
class RemoteZip(io.RawIOBase):
    def __init__(self,url,meta):self.url=url;self.size=meta['size'];self.etag=meta['etag'];self.pos=0;self.bytes_read=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        assert n<64*1024**2,'Refuse accidental full-video download'
        start=self.pos;req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+n-1}','If-Match':self.etag})
        with urllib.request.urlopen(req,timeout=25) as r:
            assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{start+n-1}/{self.size}';data=r.read(n+1)
        assert len(data)==n;self.pos+=n;self.bytes_read+=n;return data
def central_index():
    meta=head(VIDEOS);remote=RemoteZip(VIDEOS,meta)
    with zipfile.ZipFile(remote) as z:
        entries=[dict(name=x.filename,compressed=x.compress_size,size=x.file_size,crc32=x.CRC,offset=x.header_offset,method=x.compress_type) for x in z.infolist()]
    assert entries and remote.bytes_read<1024**2
    # A second remote-directory read must reproduce every entry without downloading videos.
    second=RemoteZip(VIDEOS,meta)
    with zipfile.ZipFile(second) as z:assert entries==[dict(name=x.filename,compressed=x.compress_size,size=x.file_size,crc32=x.CRC,offset=x.header_offset,method=x.compress_type) for x in z.infolist()]
    return dict(meta=meta,entries=entries,network_bytes=remote.bytes_read+second.bytes_read)
def download(url,name,maxsize):
    meta=head(url);assert meta['size']<=maxsize;dest=STORE/name
    if dest.exists():assert dest.stat().st_size==meta['size'];return dict(**meta,path=str(dest),sha256=sha(dest))
    part=dest.with_suffix(dest.suffix+'.partial')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'If-Match':meta['etag']}),timeout=25) as r,part.open('wb') as f:
        assert r.status==200
        while True:
            chunk=r.read(1024**2)
            if not chunk:break
            f.write(chunk)
    assert part.stat().st_size==meta['size'];part.replace(dest);return dict(**meta,path=str(dest),sha256=sha(dest))
def prepare():
    save(HERE/'protocol.json',dict(code_sha256=sha(__file__),hypothesis='Independent human step annotations with source RGB may resolve whether camera-derived cadence is accurate rather than merely agreeing with an optical proxy. Audit access,task/view/inputsemantics andtimealignment before any pose/modelbenchmark.',
      sources=dict(project=BASE,data=DATA,videos=VIDEOS,paper=BASE+'mattfeld2017-pedom.pdf',dissertation='https://cecas.clemson.edu/~ahoover/theses/mattfeld-diss.pdf'),
      stage='First feasibility stage only: verifiedmetadata/ZIPindex,31.7MBpublicsensor+stepannotations,originalpaper. No video extraction,pose inference,scoreprediction ornewclinicalmodel until groundtruth semantics/timealignment/reuseconditions andcamera framing checked.',
      access='ObservedofficialHTTPSdirectlinks withoutlogin/application. TLSverification retained. Public availability is not a blanket data-license/IRB determination. Inspect archive andprimarypapers for actualreuseandannotation conditions;do not declare licensing cleared from a code license. No forms/emails/agreements submitted.',
      input='30participants reported byofficialpage,3walkingactivitytypes,15Hzsensorline indices forsteps. Rawvideo task/camera/framing andsample-levelmapping not yet verified;noSPPB/TUGlabels assumed. May be moving camera anddifferent fromfixedphone target.',
      pilot='HEADdata/videos andreadZIP64centraldirectorytwice viaHTTP206/ETag/ContentRange with<1MBnetworkbytes. Recordactualvideoentrynames,sizes,CRC,offsets;do not infervideo integrity fromindex.',
      main='DownloadData.zip andoriginalpaper toD withContentLength/ETag/SHA. CRCtestalllocalDataentries;writefileinventoryandboundedtextpreviews afterexcludingidentifiers. Inventoryallvideoentriesbutdo notfetch11.7GB. Extractpapertextlocallyinanalysis;recordterms/alignmentlimits beforeselectingsamplevideo.',
      success='Feasibility gates:actualanonymousdownload;allarchiveCRCsvalid;person/activitylabelmatrixidentified;eventunits/step-vs-shift/side andvideooffsetconventiondocumented;video framing andterms checkedbeforemodelbenchmark. Partialaccessalone is not cadencevalidation orlearning-success.',
      expected=dict(pilot_remote_directory_replays=2,main_data_archives=1,main_papers=1,new_model_fits=0),
      verification='Independentmanualcentraldirectoryentrycount/signatures/CRCcomparedtozipfile,repeatmetadataindex;localDataZIPCRC/entrybytes/linecounts/monotoniceventindex andsensorbounds whereidentifiable;sourceSHAandoriginalHTTPmetadata. Keep allfiles/failedcases;neverfittemporaloffset to predictions.',
      resources='CPU1,metadata+31.7MBdata+1MBpaper onD;smallcode/inventory/reportsC. Videoarchive11.7GBnotdownloaded atthisstage. ExistingMobilise-Dscorepeople/bouts/5seeds andallotherexperimentsunchanged.',storage=dict(small=str(HERE),large=str(STORE))))
def pilot():
    q=central_index();save(HERE/'video-index.json',q);save(HERE/'pilot-verification.json',dict(passed=True,remote_directory_replays=2,entries=len(q['entries']),bytes_read=q['network_bytes'],code_sha256=sha(__file__)));print('PILOT_INDEX',len(q['entries']),q['network_bytes'],flush=True)
def main():
    p=json.loads((HERE/'protocol.json').read_text());v=json.loads((HERE/'pilot-verification.json').read_text());assert p['code_sha256']==v['code_sha256']==sha(__file__) and v['passed'];save(HERE/'main-process.json',dict(pid=os.getpid(),entry='study.py main',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()));STORE.mkdir(parents=True,exist_ok=True);started=time.time()
    data=download(DATA,'Data.zip',40*1024**2);paper=download(BASE+'mattfeld2017-pedom.pdf','mattfeld2017-pedom.pdf',3*1024**2);rows=[]
    with zipfile.ZipFile(data['path']) as z:
        assert z.testzip() is None
        for x in z.infolist():rows.append(dict(name=x.filename,size=x.file_size,compressed=x.compress_size,crc32=x.CRC))
    save(HERE/'data-index.json',dict(source=data,entries=rows));save(HERE/'main-completion.json',dict(downloads=[data,paper],data_entries=len(rows),crc_all_passed=True,new_model_fits=0,code_sha256=sha(__file__),seconds=time.time()-started));save(HERE/'main-process-exit.json',dict(pid=os.getpid(),exit_code=0));print('ACQUIRED',len(rows),data['size'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);a=ap.parse_args();{'prepare':prepare,'pilot':pilot,'main':main}[a.mode]()
