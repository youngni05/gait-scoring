"""Adapt the verified first-run executor to the explicit 695-person front-only contract."""
from pathlib import Path
import ast, hashlib, json, shutil, zipfile
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'bundle-py38'
BUNDLE=HERE/'gpu-bundle'
BUNDLE.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert json.loads((HERE/'preparation_verification.json').read_text(encoding='utf-8'))['passed']
for name in ['MB_lite.bin','MOTIONBERT_LICENSE','lib/__init__.py','lib/model/__init__.py',
             'lib/model/DSTformer.py','lib/model/drop.py']:
    p=BUNDLE/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,p)
for name in ['train_only.npz','baseline_heads.npz','baseline_predictions.npz']:
    shutil.copyfile(HERE/name,BUNDLE/name)
code=(SOURCE/'run.py').read_text(encoding='utf-8')
replacements={
    '(203,2,124,17,3)':'(1064,1,124,17,3)',
    "d['y'].shape!=(203,2)":"d['y'].shape!=(1064,2)",
    '(203,2,512)':'(1064,1,512)',
    "len(set(d['video_id']))!=203 or len(set(d['person_id']))!=175":"len(set(d['video_id']))!=1064 or len(set(d['person_id']))!=695",
    'len(te)!=35 or len(tr)!=140':'len(te)!=139 or len(tr)!=556',
    "'evaluate_windows':['front','back']":"'evaluate_windows':['front']",
    'np.empty((len(te),2,2))':'np.empty((len(te),1,2))',
    'for w in range(2):':'for w in range(1):',
}
for old,new in replacements.items():
    assert old in code,old
    code=code.replace(old,new)
tree=ast.parse(code)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
lines=code.splitlines(keepends=True)
replay='''def replay(out,device):
    d=data();rows=[]
    for fold in range(5):
        m=Scorer('head_only',fold).to(device)
        for cohort in (0,1):
            i=int(np.flatnonzero((d['fold']==fold)&(d['cohort']==cohort))[0])
            with torch.inference_mode():
                actual=m(torch.from_numpy(d['x'][i:i+1,0]).to(device)).cpu().numpy()
                expected=m(torch.from_numpy(d['features'][i:i+1,0]).to(device)).cpu().numpy()
            err=float(abs(actual-expected).max())
            if err>1e-4: raise ValueError('baseline startup score differs '+str(err))
            rows.append({'fold':fold,'cohort':cohort,'window':0,'max_score_error':err})
        del m
    bad={k:v.copy() for k,v in d.items()};p=bad['person_id'][0];idx=np.flatnonzero(bad['person_id']!=p)[0];bad['person_id'][idx]=p
    try: validate_data(bad)
    except ValueError: pass
    else: raise ValueError('corrupted identity accepted')
    try: protocol(ROOT/'protocol.draft.json')
    except ValueError: pass
    else: raise ValueError('draft training accepted')
    save(out/'replay.json',{'rows':rows,'identity_corruption_rejected':True,'draft_rejected':True,'real_data_optimizer_steps':0,'people':695,'records':1064})
    print(json.dumps({'replay_max':max(r['max_score_error'] for r in rows),'records':1064,'real_data_optimizer_steps':0}))

'''
code=''.join(lines[:node.lineno-1])+replay+''.join(lines[node.end_lineno:])
marker="    if d['x'].dtype!=np.float32: raise ValueError('unexpected input dtype')"
assert marker in code
code=code.replace(marker,marker+'''
    if d['cohort'].shape!=(1064,) or int((d['cohort']==0).sum())!=203 or int((d['cohort']==1).sum())!=861: raise ValueError('cohort mismatch')
    for k in range(5):
        for cohort,expected in ((0,35),(1,104)):
            if len(set(d['person_id'][(d['fold']==k)&(d['cohort']==cohort)]))!=expected: raise ValueError('cohort fold count mismatch')''')
(BUNDLE/'run.py').write_text(code,encoding='utf-8')
batch=(SOURCE/'batch.py').read_text(encoding='utf-8')
for old,new in {
    "return {'front_mae':person(abs(pred[:,0]-d['y']).mean(1)),'back_mae':person(abs(pred[:,1]-d['y']).mean(1)),'front_back_gap':person(abs(pred[:,0]-pred[:,1]).mean(1))}":"return {'front_mae':person(abs(pred[:,0]-d['y']).mean(1))}",
    '(203,2,2)':'(1064,1,2)','list(range(203))':'list(range(1064))',
    "'people':175,'records':203":"'people':695,'records':1064",
    'Development-only; front/back of same recording, not re-filming or view invariance. No clinical pass threshold.':'Expanded development only; front124 only; no back-window or final-test claim.',
}.items():
    assert old in batch,old
    batch=batch.replace(old,new)
(BUNDLE/'batch.py').write_text(batch,encoding='utf-8')
draft=json.loads((SOURCE/'protocol.draft.json').read_text(encoding='utf-8-sig'))
draft['evaluate_windows']=['front'];draft['seeds']=[20261028]
draft['protocol_note']='695-person fixed expansion; old fold membership preserved; no tuning on fold results.'
save(BUNDLE/'protocol.draft.json',draft)
ready=dict(draft,status='ready',steps=200,batch_size=32,head_lr=1e-4,encoder_lr=1e-5,weight_decay=0.0,checkpoint_every=10)
save(BUNDLE/'protocol.ready.json',ready)
(BUNDLE/'launch.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run inside the allocated GPU shell, not the login shell."
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p logs
exec > >(tee -a logs/expanded_run.log) 2>&1
echo "EXPANSION_PREFLIGHT_STARTED"
date -Is
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
timeout --kill-after=10s 300s python3 -u -B run.py check --device cuda --output checks/replay
timeout --kill-after=10s 600s python3 -u -B run.py profile --device cuda --batch-size 32 --output checks/profile_b32
echo "EXPANDED_TRAINING_STARTED"
date -Is
python3 -u -B batch.py run --config protocol.ready.json --output runs/main --device cuda
python3 -u -B batch.py summarize --config protocol.ready.json --output runs/main
date -Is
echo "EXPANDED_TRAINING_AND_SUMMARY_FINISHED"
''',encoding='utf-8',newline='\n')
(BUNDLE/'README.md').write_text('''# 695명 확대 GDI 부분학습

실제 입력 695명·1064기록, 앞124프레임·17관절·x/y/신뢰도만 사용한다. 기존175명과 개발520명만 포함했고 보호37/38명·최종예약223명은 포함하지 않았다.
기존175명의5fold를 보존하고 새520명은 seed20261030으로104명씩 나눴다. 매회 학습556명·평가139명이다. 입력구간을 앞/뒤로 복제하지 않았으며 앞뒤 안정성을 평가하지 않는다.
CPU에서 fold별 학습556명의 고정 특징에만 StandardScaler와 Ridge(alpha1)를 적합했다. baseline_heads는 이 확대 Ridge 초기값이며 baseline_predictions는 각 평가사람의 예측이다.
head_only/tail/last_stage 세 방식은 동일 초기값·seed20261028·batch32·200 updates·출력부lr1e-4·본체lr1e-5·weight_decay0을 쓴다. 초기175명 실험과 갱신수는 같지만 자료를 통과하는 횟수는 더 적다.
원래 모델 구조·학습 파라미터 범위·loss·중단재개·1e-4점 초기재현 기준을 유지한다. CUDA check는 fold별 기존/새 집단 각1기록 총10건을 확인한다. 합성 smoke/profile은 실제 GDI 학습이 아니다.
기존 실행 폴더와 결과를 덮어쓰지 말고 별도 gdi-motionbert-expanded695-20261030 폴더에서 실행한다. 확정된 계획을 임의로 재탐색하지 않는다.
''',encoding='utf-8')
names=['train_only.npz','baseline_heads.npz','baseline_predictions.npz','MB_lite.bin',
       'MOTIONBERT_LICENSE','run.py','batch.py','protocol.draft.json','README.md','launch.sh',
       'lib/__init__.py','lib/model/__init__.py','lib/model/DSTformer.py','lib/model/drop.py']
save(BUNDLE/'manifest.json',{n:sha(BUNDLE/n) for n in names})
for p in BUNDLE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
archive=HERE/'gdi-expanded695.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in names+['manifest.json','protocol.ready.json']:
        z.write(BUNDLE/name,'gdi-motionbert-expanded695-20261030/'+name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in names:
        assert hashlib.sha256(z.read('gdi-motionbert-expanded695-20261030/'+name)).hexdigest()==sha(BUNDLE/name)
save(HERE/'gpu_bundle_verification.json',{'archive_sha256':sha(archive),'bytes':archive.stat().st_size,
    'python38_syntax_passed':True,'zip_integrity_passed':True,'remote_cuda_check_pending':True,
    'data_shape':[1064,1,124,17,3],'people':695,'protected_data_included':False})
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}))
