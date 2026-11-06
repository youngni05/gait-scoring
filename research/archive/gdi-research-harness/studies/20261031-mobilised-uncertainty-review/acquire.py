from common import *
import urllib.request,urllib.parse,zipfile
FILES={'Data Dictionary.zip':'20c43429f655575c6868ea2ef8343176','Clinical Data.zip':'3b3d8305fa7ef6923341a85bfd4a3945','Aggregated DMO Data.zip':'b6497e3cdcad2166f3ba38d5d3d64e35','Flowcharts.zip':'9ae62e54252a0e8ec0e0b44eba2340a6'}
FILES['Main datasets for analysis.zip']='6b28660706b791d9a4afa2a718f9389b'
def main():
    receipts=[]
    for name,md5 in FILES.items():
        url='https://zenodo.org/records/21060991/files/'+urllib.parse.quote(name)+'?download=1'
        p=HERE/'raw'/name;p.parent.mkdir(exist_ok=True)
        if not p.exists():
            with urllib.request.urlopen(url,timeout=90) as r:b=r.read()
            assert hashlib.md5(b).hexdigest()==md5;p.write_bytes(b)
        assert hashlib.md5(p.read_bytes()).hexdigest()==md5
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            names=z.namelist()
            for n in names:
                target=(HERE/'extracted'/n).resolve()
                assert target.is_relative_to((HERE/'extracted').resolve())
                if not n.endswith('/'):
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
        receipts.append(dict(name=name,url=url,bytes=p.stat().st_size,md5=md5,sha256=sha(p),files=names))
        save('acquisition.json',receipts);print(name,p.stat().st_size,'entries',len(names),flush=True)
if __name__=='__main__':main()
