import sys,io,tarfile,json
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import numpy as np
from scipy.io import loadmat
root=Path('D:/gait-research-data/MoVi/raw')
with tarfile.open(root/'F_Subjects_1_45.tar') as t:
    f=next(v for v in t if Path(v.name).name=='F_v3d_Subject_1.mat')
    m=loadmat(io.BytesIO(t.extractfile(f).read()),simplify_cells=True)['Subject_1_F']['move']
for k in ['segmentName','segmentParent','segmentDescription','segmentType','lcData']:
    x=m[k];print(k,type(x),str(x)[:5500])
names=list(m['markerName']);segs=list(m['segmentName']);f=(1484-1)*4
print('markers',[(n,m['markerLocation'][f,names.index(n)].round(2).tolist()) for n in ['LFWT','RFWT','LBWT','RBWT','LHIP','LHJC','LKJC','LAJC','LHEE','LTOE','LMT1','LMT5']])
for n in ['RPV','LTH','LSK','LFT']:
    i=segs.index(n);print(n,'aff',m['jointsAffine_v3d'][f,i].round(5));print('exp',m['jointsExpMaps_v3d'][f,i]);print('translation',m['jointsTranslation_v3d'][f,i])
global_aff={}
def global_matrix(i):
    if i not in global_aff:
        par=int(m['segmentParent'][i])-1
        local=m['jointsAffine_v3d'][f,i].copy();local[:3,:3]=local[:3,:3].T
        global_aff[i]=local if par<0 else global_matrix(par)@local
    return global_aff[i]
for n in ['RPV','LTH','LSK','LFT']:
    g=global_matrix(segs.index(n));print(n,'GLOBAL',g.round(3))
    for marker in ['LHIP','LKJC','LAJC','LHEE','LTOE']:
        print(marker,((m['markerLocation'][f,names.index(marker)]-g[:3,3])@g[:3,:3]).round(2))
ref=np.load('outputs/gdi-reference-investigation-20261030/gdi_original_active166_reference.npz')
print('reference',ref.files)
for i,n in enumerate(ref['angle_names']):
    x=ref['control_angle_curves'][:,:,i];print(n,'mean/min/max',x.mean(),x.mean(0).min(),x.mean(0).max(),'phase0,50,60,100',x.mean(0)[[0,25,30,50]])
