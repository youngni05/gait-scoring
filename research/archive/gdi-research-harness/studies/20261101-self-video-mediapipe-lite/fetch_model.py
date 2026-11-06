"""Download only the official Lite model, bounded in size, with provenance."""
import urllib.request
import datetime
import zipfile
from common import *
URL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'

def main():
    BIG.mkdir(parents=True,exist_ok=True)
    assert not TASK.exists(),'Do not replace a previously pinned model'
    with urllib.request.urlopen(URL,timeout=60) as response:
        payload=response.read(15_000_001)
        assert 0<len(payload)<=15_000_000
        headers=dict(response.headers.items());status=response.status
    assert status==200
    TASK.write_bytes(payload)
    with zipfile.ZipFile(TASK) as z:
        files=[dict(name=n,bytes=z.getinfo(n).file_size) for n in z.namelist()]
        assert z.testzip() is None
    save(HERE/'model-receipt.json',dict(url=URL,downloaded_at=datetime.datetime.now().astimezone().isoformat(),status=status,
        bytes=len(payload),sha256=sha(TASK),headers=headers,archive_files=files))
    preserve={}
    for path in FULL.iterdir():
        if path.is_file():preserve[path.name]=sha(path)
    save(HERE/'full-preservation.json',preserve)
    print('LITE',len(payload),sha(TASK),files,flush=True)

if __name__=='__main__':main()
