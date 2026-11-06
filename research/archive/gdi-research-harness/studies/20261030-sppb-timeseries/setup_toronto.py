"""Create an isolated real-TUG follow-up using already audited source input."""
from common import *
import openpyxl
from sklearn.model_selection import KFold

def run():
    dest = HERE.parent/'20261030-tug-toronto-timeseries'
    dest.mkdir(exist_ok=True)
    old = HERE.parent/'20261030-toronto-poma-tug'
    source = ROOT/'outputs/gait-public-access-20261015/Toronto/Table 1.xlsx'
    wb = openpyxl.load_workbook(source,data_only=True)
    labels={int(r[0]):float(r[9]) for r in list(wb.active.values)[1:] if isinstance(r[0],int)}
    wb.close()
    raw=np.load(old/'inputs.npz',allow_pickle=False)
    ids=raw['person'].astype(str);people=np.unique(ids)
    folded={}
    for fold,(_,ix) in enumerate(KFold(5,shuffle=True,random_state=20261030).split(people)):
        folded.update({p:fold for p in people[ix]})
    x=raw['input'].reshape(59,124,51).transpose(0,2,1)
    y=np.array([labels[int(p)] for p in ids],dtype=np.float32)
    np.savez_compressed(dest/'input.npz',x=x,y=y,ids=ids,folds=np.array([folded[p] for p in ids]),frames=np.full(len(x),124,dtype=np.float32))
    protocol={
        'declared_before_fitting':True,'question':'Predict clinical participant TUG seconds from walking-only video-derived skeleton clips in unseen people',
        'sources':{'input':str(old/'inputs.npz'),'input_sha256':sha(old/'inputs.npz'),'clinical_table':str(source),'clinical_sha256':sha(source),'source_preparation_sha256':sha(old/'prepare.py')},
        'input':'59 original audited Alphapose windows; 12 full participant IDs; each124frames at30Hz (4.1 seconds), existing COCO to H36M17-joint normalization; XY and confidence only',
        'target':'Table1 column10 TUG(s), participant ID exact match; no score conversion; same person label reused across walking clips (weak supervision, not59 independently measured TUG labels)',
        'split':'label-independent person5fold seed20261030, nested training-person3fold seed20261031; all clips for a person stay together; no augmentation or external pretraining',
        'weight':'equal total per person in training; average clip predictions before person MAE/RMSE',
        'cpu':'Mean/Median constants; Ridge alpha[1,100]; RF128trees leaf[2,5]; 153summaryfeatures = channel mean/std/mean absolute temporal difference; scale trained on training people only',
        'gpu':'CNN vs TCN,3layers width24,kernel5,dilation[1,1,1]vs[1,2,4],dropout.2,AdamW lr.001 decay.001,person-weighted standardized MSE; train-only normalization; inner3fold best epoch median refit; max80epochs patience12; one seed20261030',
        'scalar_input':'constant log1p124 for compatibility; carries no target/person information, never score or participant metadata',
        'pilot':'3epochs on fold0 inner-training people; realCUDA calculation/finite loss+prediction/checkpoint reload equivalence/time+GPU peak memory; no outer evaluation people used for pilot',
        'scope':'small feasibility comparison; 12participants cannot establish clinical validity or reliable large-network advantage; frozen MotionBERT prior experiment preserved; no GDI input used',
        'licence':'Toronto source Figshare release CC0, independently rechecked canonical source page',
        'no_clipping_or_rounding':True}
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (dest/'input-audit.json').write_text(json.dumps({'input_sha256':sha(dest/'input.npz'),'protocol_sha256':sha(dest/'protocol.json'),'people':len(people),'clips':len(x),'target_range':[float(y.min()),float(y.max())],'source_labels_exact':True},indent=2))
    common=(HERE/'common.py').read_text(encoding='utf8')
    begin=common.index('def features(');end=common.index('def metrics(',begin)
    common=common[:begin]+'''def features(x, frames):
    return np.concatenate([x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)],axis=1)

'''+common[end:]
    begin=common.index('def mae(');end=common.index('def median(',begin)
    common=common[:begin]+'''def mae(y, pred, ids):
    return float(np.mean([abs(np.mean(y[ids==p])-np.mean(pred[ids==p])) for p in np.unique(ids)]))

'''+common[end:]
    common=common[:common.index('def metrics(')]+'''def metrics(y, pred, ids):
    errors=np.array([np.mean(y[ids==p])-np.mean(pred[ids==p]) for p in np.unique(ids)])
    return {'person_MAE':float(np.mean(np.abs(errors))),'person_RMSE':float(np.sqrt(np.mean(errors**2)))}
'''
    (dest/'common.py').write_text(common,encoding='utf8')
    cpu=(HERE/'cpu.py').read_text(encoding='utf8').replace("('FramesRidge',[1.,100.]),",'').replace("'FramesRidge',",'')
    (dest/'cpu.py').write_text(cpu,encoding='utf8')
    gpu=(HERE/'gpu.py').read_text(encoding='utf8').replace('layers=[];cin=54','layers=[];cin=51').replace('SPPB_PILOT_PASSED','TORONTO_TUG_PILOT_PASSED').replace('SPPB_TIMESERIES_TRAINING_FINISHED','TORONTO_TUG_TRAINING_FINISHED')
    (dest/'gpu.py').write_text(gpu,encoding='utf8')
    (dest/'submit.slurm').write_text((HERE/'submit.slurm').read_text().replace('sppb-walk-ts','tug-toronto-ts'),encoding='utf8',newline='\n')
    print(dest)
    print((dest/'input-audit.json').read_text())

if __name__=='__main__':run()
