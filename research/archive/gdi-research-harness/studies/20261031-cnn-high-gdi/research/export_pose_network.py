"""Export the unchanged official BODY25 network graph and weights from OpenCV."""
from pathlib import Path
import re, json, hashlib
import cv2, numpy as np
HERE=Path(__file__).resolve().parent
OP=HERE.parents[3]/'tmp/openpose-cpu/openpose/models/pose/body_25'
proto=OP/'pose_deploy.prototxt'; weights=OP/'pose_iter_584000.caffemodel'
assert hashlib.md5(weights.read_bytes()).hexdigest()=='78287b57cf85fa89c03f1393d368e5b7'
net=cv2.dnn.readNetFromCaffe(str(proto),str(weights))
graph=[]; arrays={}
for i,s in enumerate(proto.read_text().split('layer {')[1:]):
    def texts(key): return re.findall(r'\b'+key+r':\s*"([^"]+)"',s)
    def number(key,default):
        m=re.search(r'\b'+key+r':\s*(\d+)',s)
        return int(m[1]) if m else default
    name=texts('name')[0]; kind=texts('type')[0]
    assert kind in ['Convolution','PReLU','ReLU','Pooling','Concat']
    row=dict(name=name,type=kind,bottom=texts('bottom'),top=texts('top')[0],
             kernel=number('kernel_size',1),stride=number('stride',1),pad=number('pad',0),axis=number('axis',1))
    blobs=net.getLayer(net.getLayerId(name)).blobs
    row['weights']=[]
    for j,b in enumerate(blobs):
        key='layer%d_blob%d'%(i,j);arrays[key]=b;row['weights'].append(key)
    graph.append(row)
(HERE/'pose_probe/network.json').write_text(json.dumps(graph,indent=2))
np.savez(str(HERE/'pose_probe/network_weights.npz'),**arrays)
print('layers',len(graph),'parameters',sum(x.size for x in arrays.values()))
