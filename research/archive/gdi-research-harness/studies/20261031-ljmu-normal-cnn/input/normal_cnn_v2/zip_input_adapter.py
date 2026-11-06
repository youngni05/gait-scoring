"""Read-only ZIP input adapter for the unchanged teammate read_record function.

Only the filesystem interface and file-hash helper are adapted. JSON decoding,
person selection, validation, feature extraction, windowing, and outputs remain
the supplied functions. One member's decompressed bytes are cached at a time.
"""
import hashlib,zipfile
from collections import defaultdict
from pathlib import Path,PurePosixPath

class ZipMember:
    def __init__(self,reader,info):self.reader=reader;self.info=info;self.name=PurePosixPath(info.filename).name
    def read_bytes(self):
        if self.reader.cached_name!=self.info.filename:
            self.reader.cached_data=self.reader.archive.read(self.info)
            self.reader.cached_name=self.info.filename
        return self.reader.cached_data
    def read_text(self,encoding='utf-8'):return self.read_bytes().decode(encoding)

class ZipFolder:
    def __init__(self,reader,key):self.reader=reader;self.key=key
    def glob(self,pattern):
        if pattern!='*_keypoints.json':raise ValueError('Unsupported glob: '+pattern)
        return (ZipMember(self.reader,x) for x in self.reader.index.get(self.key,[]))

class ZipInput:
    def __init__(self,archive,filesystem_base,extracted_root=None,archive_prefix=''):
        self.archive_path=Path(archive).resolve();self.archive=zipfile.ZipFile(self.archive_path)
        self.base=Path(filesystem_base);self.extracted_root=Path(extracted_root).resolve() if extracted_root else None
        self.prefix=archive_prefix.rstrip('/');self.index=defaultdict(list);self.cached_name=None;self.cached_data=b''
        for info in self.archive.infolist():
            if not info.is_dir() and info.filename.endswith('_keypoints.json'):
                self.index[str(PurePosixPath(info.filename).parent)].append(info)
    def __truediv__(self,value):
        p=Path(value)
        # Selection CSVs remain ordinary small files next to the input manifest.
        if p.suffix.lower()=='.csv':return self.base/p
        if p.is_absolute():
            if self.extracted_root is None:raise ValueError('Absolute folder requires extracted_root mapping')
            rel=p.resolve().relative_to(self.extracted_root).as_posix()
        else:rel=p.as_posix()
        key=(self.prefix+'/' if self.prefix else '')+rel
        return ZipFolder(self,key)
    def close(self):self.archive.close()

def install_sha_adapter(module):
    original=module.sha
    def sha(value):
        if isinstance(value,ZipMember):return hashlib.sha256(value.read_bytes()).hexdigest()
        return original(value)
    module.sha=sha
    return original
