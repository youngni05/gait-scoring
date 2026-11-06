"""Independent known-pose, rigid-coordinate and original-workbook checks."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from analyze import HERE,nine_angles,score,REF
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'D:/Projects/D_P/outputs/gdi-reference-investigation-20261030/python-libs')
import xlrd

def synthetic(side,tilt=0,obliquity=0,pelvis_rotation=0,hipflex=0,hipadd=0,hiprot=0,knee=0,ankle=0,foot_rotation=0,heading=0):
    s=1 if side=='L' else -1
    H=Rot.from_euler('Z',heading,degrees=True).as_matrix();P=H@Rot.from_euler('YXZ',[tilt,s*obliquity,-s*pelvis_rotation],degrees=True).as_matrix()
    T=P@Rot.from_euler('YXZ',[-hipflex,-s*hipadd,-s*hiprot],degrees=True).as_matrix();S=T@Rot.from_euler('Y',knee,degrees=True).as_matrix();F=S@Rot.from_euler('Y',-ankle,degrees=True).as_matrix()@Rot.from_euler('Z',-s*foot_rotation,degrees=True).as_matrix()
    center=np.array([400.,200.,1000.]);hip=center+P@np.array([0.,s*100.,0.]);kj=hip-T[:,2]*400;aj=kj-S[:,2]*400
    vals={'LFWT':center+P@np.array([80,100,0]),'RFWT':center+P@np.array([80,-100,0]),'LBWT':center+P@np.array([-80,100,0]),'RBWT':center+P@np.array([-80,-100,0]),side+'HIP':hip,side+'KJC':kj,side+'AJC':aj,side+'KNE':kj+s*50*T[:,1],side+'KNEIN':kj-s*50*T[:,1],side+'TOE':aj+F[:,0]*200,side+'HEE':aj-F[:,0]*50,side+'MT1':aj+F[:,0]*150-s*50*F[:,1],side+'MT5':aj+F[:,0]*150+s*50*F[:,1]}
    p=lambda n:np.tile(vals[n],(3,1));angles,alt,orth=nine_angles(p,side,H[:,0]);return angles[0],orth

cases=[]
for side in ['L','R']:
    for parameters in [{},{'hipflex':30},{'hipadd':10},{'hiprot':15},{'knee':40},{'ankle':10},{'obliquity':5},{'pelvis_rotation':7},{'foot_rotation':12},{'tilt':12,'hipflex':30,'hipadd':8,'hiprot':6,'knee':40,'ankle':10}]:
        x,orth=synthetic(side,**parameters);expected=np.array([parameters.get('tilt',0),parameters.get('obliquity',0),parameters.get('pelvis_rotation',0),parameters.get('hipflex',0),parameters.get('hipadd',0),parameters.get('hiprot',0),parameters.get('knee',0),parameters.get('ankle',0)])
        err=float(np.max(np.abs(x[:8]-expected)));assert err<1e-9,(side,parameters,x,expected)
        if len(parameters)<=1:
            foot_expected=parameters.get('foot_rotation',parameters.get('pelvis_rotation',parameters.get('hiprot',0)))
            assert abs(x[8]-foot_expected)<1e-9,(side,parameters,x[8],foot_expected)
        y,_=synthetic(side,heading=73,**parameters);rigid=float(np.max(np.abs(x-y)));assert rigid<1e-9
        cases.append({'side':side,'parameters':parameters,'max_error_deg':err,'heading_invariance_error_deg':rigid,'orthogonality_max_error':orth})

source=Path('outputs/gdi-reference-investigation-20261030/publisher-supplement.xls');book=xlrd.open_workbook(source)
s=book.sheet_by_name('GDI Calculations');b=np.array([[s.cell_value(r,c) for c in range(15)] for r in range(2,461)]);example=np.array(s.col_values(18,2,461));cached=s.cell_value(11,25)
co=book.sheet_by_name('control kinematics');controls=np.array([[co.cell_value(r,c) for r in range(1,460)] for c in range(1,167)])
features=np.array([[sum(float(x[k])*float(b[k,j]) for k in range(459)) for j in range(15)] for x in controls]);center=features.mean(0);logs=np.log(np.linalg.norm(features-center,axis=1));mean=logs.mean();sd=logs.std(ddof=1)
exfeatures=np.array([sum(float(example[k])*float(b[k,j]) for k in range(459)) for j in range(15)]);rep=100-10*(np.log(np.linalg.norm(exfeatures-center))-mean)/sd
assert abs(rep-cached)<1e-8
data=np.load(HERE/'angle-curves.npz');curves=data['curves'];cycles=json.loads((HERE/'cycles.json').read_text());accepted=[c for c in cycles if c['accepted']]
errors=[]
for c in accepted:
    curve=curves[c['curve_index']];vec=np.concatenate([curve[:,j] for j in range(9)]);f=np.array([sum(float(vec[k])*float(b[k,j]) for k in range(459)) for j in range(15)]);g=100-10*(np.log(np.linalg.norm(f-center))-mean)/sd
    errors.append(abs(g-c['exploratory_gdi']))
assert max(errors)<1e-8
subjects=json.loads((HERE/'subjects.json').read_text());assert len(subjects)==28 and len(curves)==len(accepted)==52
for c in accepted:
    p=next(p for p in subjects if p['id']==c['id']);assert p['mocap_start_one_based']<=c['start_mocap_one_based']<c['end_mocap_one_based']<=p['mocap_end_one_based'];assert np.isfinite(curves[c['curve_index']]).all()
report={'numerical_checks_passed':True,'synthetic_known_pose_cases':len(cases),'synthetic_details':cases,'original_workbook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_example_gdi':float(rep),'original_example_cached':cached,'original_example_error':float(abs(rep-cached)),'independent_scalar_score_replay_max_error':float(max(errors)),'cycles_replayed':len(curves),'people':len(subjects),'clinical_angle_ground_truth_available':False,'clinical_gdi_equivalence_passed':False,'training_labels_accepted':0}
(HERE/'math-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='synthetic_details'}))
