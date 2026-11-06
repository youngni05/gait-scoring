from common import *
import zipfile,xml.etree.ElementTree as ET
def workbook_cells(path):
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        ss=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        rel={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        out={}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets',ns):
            p=rel[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            p=p.lstrip('/') if p.startswith('/') else 'xl/'+p
            rows=[]
            for row in ET.fromstring(z.read(p)).findall('.//m:sheetData/m:row',ns):
                cells={}
                for c in row.findall('m:c',ns):
                    v=c.find('m:v',ns)
                    if v is not None:cells[c.attrib['r']]=ss[int(v.text)] if c.attrib.get('t')=='s' else v.text
                    elif c.attrib.get('t')=='inlineStr':cells[c.attrib['r']]=''.join(c.find('m:is',ns).itertext())
                if cells:rows.append(cells)
            out[sheet.attrib['name']]=rows
    return out
def main():
    p=HERE/'extracted/Data Dictionary/CVS Data Dictionary Final Release.xlsx';sheets=workbook_cells(p);save('dictionary-cells.json',sheets)
    print('SHEETS',[(k,len(v)) for k,v in sheets.items()])
    for sheet,rows in sheets.items():
        hits=[r for r in rows if any(any(s in str(v).lower() for s in ['tugtime','5crtresult','sppb_total','sppbtotal','validday','wb_10','walking speed']) for v in r.values())]
        if hits:print('HITS',sheet,json.dumps(hits,ensure_ascii=True)[:8000])
    for p in (HERE/'extracted/Aggregated DMO Data').glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            print(p.name,[(n,z.getinfo(n).file_size) for n in z.namelist()])
            for name in z.namelist():
                target=(HERE/'extracted/Aggregated'/name).resolve();assert target.is_relative_to((HERE/'extracted/Aggregated').resolve())
                if not name.endswith('/'):
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
                    if name.lower().endswith('.csv'):print('CSV',name,z.read(name)[:900])
if __name__=='__main__':main()
