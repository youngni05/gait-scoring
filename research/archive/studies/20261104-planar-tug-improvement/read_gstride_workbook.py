"""Read-only source extraction using bundled Python and standard XLSX XML."""
from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json
SRC=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride/GSTRIDE_DDBB.xlsx')
DEST=SRC.parent
def main():
    ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(SRC) as z:
        ss=ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings=[''.join(n.itertext()) for n in ss]
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        names=[s.attrib['name'] for s in wb.find('x:sheets',ns)]
        result={}
        for i,name in enumerate(names,1):
            root=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'));rows=[]
            for row in root.findall('.//x:sheetData/x:row',ns):
                cells={}
                for c in row:
                    v=c.find('x:v',ns)
                    if v is None:continue
                    val=v.text
                    if c.attrib.get('t')=='s':val=strings[int(val)]
                    elif c.attrib.get('t')!='str':
                        try:val=float(val)
                        except (TypeError,ValueError):pass
                    cells[c.attrib['r']]=val
                if cells:rows.append(cells)
            result[name]=rows
    (DEST/'workbook-cells.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({n:{'rows':len(r),'first_rows':r[:3]} for n,r in result.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
