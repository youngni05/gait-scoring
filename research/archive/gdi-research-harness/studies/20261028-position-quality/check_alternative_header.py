import csv,json,struct,zlib
from analyze_quality import HERE,H,ROOT,load,save

def main():
    p=ROOT/'outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip'
    st=p.stat();expected=load(H/'jobs/native124-cache-v1/config.json')['archive_stat']
    assert {'size':st.st_size,'mtime_ns':st.st_mtime_ns}==expected
    member=next(m for m in load(ROOT/'outputs/stanford-sample-check-20261016/archive-index.json')['non_json_members'] if m['name']=='annotations/alldata.csv')
    assert member['method']==8 and not member['flags']&1
    raw=bytearray();read_bytes=0;pending=b'';decoder=zlib.decompressobj(-15)
    with p.open('rb') as f:
        f.seek(member['offset']);header=f.read(30)
        assert header[:4]==b'PK\x03\x04'
        nl,el=struct.unpack_from('<HH',header,26)
        assert f.read(nl).decode()==member['name'];f.seek(el,1)
        while not raw.endswith(b'\n'):
            assert len(raw)<65536 and read_bytes<member['compressed_size']
            if not pending:pending=f.read(1);read_bytes+=1
            raw.extend(decoder.decompress(pending,1));pending=decoder.unconsumed_tail
    fields=next(csv.reader([raw.decode('utf-8-sig').strip()]))
    found=[f for f in fields if any(k in f.lower() for k in ('speed','cadence','knee','gmfcs','exam','patient'))]
    result=dict(member=member['name'],header_fields=fields,relevant_fields=found,
        compressed_bytes_read=read_bytes,decompressed_bytes=len(raw),data_rows_decompressed=0,
        data_rows_parsed=0,full_member_crc_checked=False,archive_stat_unchanged=p.stat().st_mtime_ns==st.st_mtime_ns,
        target_linkage_or_missingness_checked=False)
    save('alternative_header.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='header_fields'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
