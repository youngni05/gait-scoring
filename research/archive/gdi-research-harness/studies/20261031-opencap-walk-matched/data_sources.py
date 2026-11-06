from common import *
import urllib.request,zipfile,xml.etree.ElementTree as ET,hashlib

SOURCES={
 'Gait2Care_DDBB.xlsx':'https://zenodo.org/records/17017009/files/Gait2Care_DDBB.xlsx?download=1',
 'plos-s002.zip':'https://journals.plos.org/digitalhealth/article/file?id=10.1371%2Fjournal.pdig.0001172.s002&type=supplementary',
}
def main():
    out={}
    for name,url in SOURCES.items():
        p=HERE/'external'/name;p.parent.mkdir(exist_ok=True)
        try:
            if not p.exists():
                with urllib.request.urlopen(url,timeout=60) as r:b=r.read()
                assert b[:2]==b'PK';p.write_bytes(b)
            out[name]={'url':url,'sha256':sha(p),'bytes':p.stat().st_size,'md5':hashlib.md5(p.read_bytes()).hexdigest(),'files':zipfile.ZipFile(p).namelist()}
        except Exception as e:out[name]={'url':url,'error':str(e)}
        save('external-acquisition.json',out);print(name,{k:v for k,v in out[name].items() if k!='files'},flush=True)
    # Direct XML read preserves raw source workbook and cached source values.
    p=HERE/'external/Gait2Care_DDBB.xlsx'
    if p.exists():
        ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        with zipfile.ZipFile(p) as z:
            strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
            sheets=ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',ns)
            rel={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            tables={}
            for sh in sheets:
                target=rel[sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
                target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
                rows=[]
                for row in ET.fromstring(z.read(target)).findall('.//m:sheetData/m:row',ns):
                    cells={}
                    for c in row.findall('m:c',ns):
                        v=c.find('m:v',ns)
                        if v is not None:cells[c.attrib['r']]=strings[int(v.text)] if c.attrib.get('t')=='s' else v.text
                    if cells:rows.append(cells)
                tables[sh.attrib['name']]=rows
            save('external/gait2care-cells.json',tables)
            print('SHEETS',[(k,len(v)) for k,v in tables.items()],flush=True)

if __name__=='__main__':main()
