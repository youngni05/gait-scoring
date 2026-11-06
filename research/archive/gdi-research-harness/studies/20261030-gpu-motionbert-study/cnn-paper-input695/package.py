from pathlib import Path
import ast,hashlib,json,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent
B=HERE/'bundle'
OLD=HERE.parent/'cnn-followup695'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    with np.load(B/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files if k!='x'}
    idx=d['original_row_index']
    with np.load(OLD/'bundle/data.npz',allow_pickle=False) as z:
        for k in ('person_id','video_id','fold','cohort','y'):assert np.array_equal(z[k][idx],d[k])
    refs={}
    with np.load(OLD/'bundle/reference_predictions.npz',allow_pickle=False) as z:
        for k in z.files:refs[k]=z[k][idx].copy()
    with np.load(OLD/'analysis/cnn_oof.npz',allow_pickle=False) as z:
        for k in ('person_id','video_id','fold','y'):assert np.array_equal(z[k][idx],d[k])
        refs['previous_cnn']=z['predictions'][idx,0].copy()
    def mae(p):
        e=np.abs(p-d['y']).mean(1)
        return float(np.mean([e[d['person_id']==pid].mean() for pid in np.unique(d['person_id'])]))
    metrics={k+'_mae':mae(v) for k,v in refs.items()}
    metrics.update(people=692,records=1061,compared_same_records_and_people=True,reference_only_not_training_input=True)
    save(B/'reference_metrics.json',metrics)
    np.savez_compressed(B/'reference_predictions.npz',**refs)
    contract=json.loads((HERE/'input_audit.json').read_text());contract.update(
        axes=['record','left_right','time124','feature12'],
        features=['ankle_x','ankle_y','knee_x','knee_y','hip_x','hip_y','big_toe_x','big_toe_y',
                  'ankle_knee_hip_angle_radians','toe_ankle_knee_angle_radians','toe_ankle_distance','other_ankle_x_minus_this_ankle_x'],
        coordinate_scale='mean left/right thigh length, pelvis centered, all processed frames',
        preprocessing_context='full source-converter clip up to1499frames; selected output is same124frames',
        source_orientation='swap_orientation=False in public doublesided notebook; no side mirror added',
        source_missingness='zero/NaN xy coordinates, max per required coordinate <=25pct in selected window',
        ambiguity_policy='multiple left-half detections kept missing, unlike source accidental last-person overwrite',
        multiwindow_training_implemented=False,original_695_fold_ids_retained=True)
    save(B/'input_contract.json',contract)
    for p in B.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    for p in list(B.glob('*.sh'))+list(B.glob('*.slurm')):p.write_bytes(p.read_text().replace('\r\n','\n').encode())
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.name!='manifest.json' and not any(k in p.parts for k in ('__pycache__','checks','runs','logs'))]
    manifest={p.relative_to(B).as_posix():sha(p) for p in files};save(B/'manifest.json',manifest)
    dest=HERE/'gdi-cnn-paper-input.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,'gdi-cnn-paper-input-20261030/'+p.relative_to(B).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read('gdi-cnn-paper-input-20261030/'+n)).hexdigest()==h
    save(HERE/'package_verification.json',{'passed':True,'zip_sha256':sha(dest),'zip_bytes':dest.stat().st_size,
         'bundle_sha256':sha(B/'manifest.json'),'files':len(manifest)+1,'reference_metrics':metrics})
    print(json.dumps({'bytes':dest.stat().st_size,'references':metrics}))
if __name__=='__main__':main()
