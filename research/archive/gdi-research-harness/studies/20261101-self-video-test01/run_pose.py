"""Reuse the prior exact upload/GPU/official-decoder pipeline in isolated Test01 paths."""
import sys,importlib.util,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261101-self-video-gdi';BIG=Path('D:/gait-research-data/D_P/self-video-test01-20261101')
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ['deploy','decode']
    source=OLD/('deploy_pose.py' if mode=='deploy' else 'decode_pose.py')
    spec=importlib.util.spec_from_file_location('prior_pose_pipeline',str(source));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.HERE=HERE;m.BIG=BIG
    if mode=='deploy':m.REMOTE='/abr/RESEARCH_USER/self-video-test01-20261101'
    receipt=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),mode=mode,changed_only='HERE/BIG/REMOTE output roots; original computational code reused')
    (HERE/('reuse-'+mode+'.json')).write_text(json.dumps(receipt,indent=2),encoding='utf8')
    m.main()
