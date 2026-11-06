"""Copy the frozen CNN baseline; change only batch grouping and loss aggregation."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
H=Path(__file__).resolve().parent;B=H/'bundle';SOURCE=H.parent/'cnn-multiwindow695/bundle'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def main():
    B.mkdir(exist_ok=True)
    original=json.loads((SOURCE/'manifest.json').read_text())
    assert all(sha(SOURCE/n)==h for n,h in original.items())
    for name in ('data.npz','input_contract.json','reference_metrics.json','model.py','windows.py'):
        shutil.copyfile(SOURCE/name,B/name)
    shutil.copytree(SOURCE/'sources',B/'sources',dirs_exist_ok=True)
    for name in ('record_loss.py','audit.py','verify.py','review_pilot.py','launch.sh','submit.slurm','README.md'):
        shutil.copyfile(H/name,B/name)
    c=json.loads((SOURCE/'protocol.json').read_text())
    c.update(name='cnn_record_loss_comparison',stage='record_loss_matched_batch_no_search',batch_records=2,
        variants=['window_control','record_mean'],loss='per_record_mean_window_MSE_or_MSE_of_mean_prediction_plus_same_L2',
        baseline_batch_size=c['batch_size'],note='Same frozen CNN data, topology, optimizer, epochs, split, evaluation. Packed batches differ from historical CNN; matched window_control isolates loss order.',
        historical_baseline_manifest_sha256=sha(SOURCE/'manifest.json'))
    del c['batch_size']
    save(B/'protocol.json',c)
    contract=json.loads((B/'input_contract.json').read_text())
    contract.update(loss=c['loss'],batch_normalization='All real window-sides in two recordings per batch; identical between variants. No padding.',not_compute_matched_to_historical=True,matched_pair=True)
    save(B/'input_contract.json',contract)
    s=(SOURCE/'run.py').read_text()
    s=replace(s,'from windows import WindowSet, weighted_mse','from windows import WindowSet, weighted_mse\nfrom record_loss import packed, recording_step')
    s=replace(s,"c['stage'] == 'paper_feature_multiwindow_no_search'","c['stage'] == 'record_loss_matched_batch_no_search'")
    s=s.replace("c['batch_size']","c['batch_records']")
    s=replace(s,"('batch_size', 'max_epochs'","('batch_records', 'max_epochs'")
    a=s.index('    yt = ((torch.from_numpy(y[x.owner]');b=s.index('    vx = valid_x',a)
    s=s[:a]+"    yt = (torch.from_numpy(y.astype(np.float32)) - lo) / span\n"+s[b:]
    s=replace(s,'            totals += np.asarray(step(model, opt, train_examples[idx].to(device), yt[idx].to(device), c, example_weights[idx].to(device))) * len(idx)',
        '            values, counts = packed(x, idx, device)\n            totals += np.asarray(recording_step(model, opt, values, yt[idx].to(device), counts, c)) * len(idx)')
    # Per-epoch training/validation predictions support independent calibration audit.
    anchor="        row['seconds'] = time.perf_counter() - started"
    s=replace(s,anchor,"""        rawdir=folder/'raw_predictions';rawdir.mkdir(exist_ok=True)
        rawfile=rawdir/('epoch%04d.npz'%(epoch+1))
        np.savez_compressed(rawfile,train_raw=train_raw,validation_raw=np.empty((0,2)) if vx is None else val_raw)
        row['raw_predictions_sha256']=sha(rawfile)
"""+anchor)
    s=replace(s,"'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs, 'fit_seed': seed,","'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs, 'fit_seed': seed,\n            'loss_variant':c['loss_variant'], 'batch_records':c['batch_records'],")
    # Reject changes to source arrays/config on checkpoint resume, including synthetic tests.
    s=replace(s,"    folder.mkdir(parents=True, exist_ok=True)\n    if (folder / 'identity.json').exists():","""    identity=dict(identity,training_config=c,train_x_sha=hashlib.sha256(x.values.tobytes()+x.owner.tobytes()).hexdigest(),
        train_y_sha=hashlib.sha256(y.tobytes()).hexdigest(),train_ids_sha=hashlib.sha256(people.tobytes()).hexdigest())
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'identity.json').exists():""")
    # Completion identity retains static identity_for, while fit identity includes train fingerprints.
    s=replace(s,"require(completed['identity'] == identity_for(d, c, f, inner, c['max_epochs'], 'inner_epoch_selection'), 'Inner fit identity mismatch')",
        "require(all(completed['identity'].get(k)==v for k,v in identity_for(d, c, f, inner, c['max_epochs'], 'inner_epoch_selection').items()), 'Inner fit identity mismatch')")
    s=replace(s,'def summarize(d, c, out):','def summarize(d, c, out):\n    from audit import audit_fit\n    this = sys.modules[__name__]')
    s=replace(s,"            histories.append(load(inner_dir / 'curve.json'))","            histories.append(audit_fit(this,d,c,f,inner,inner_dir,c['max_epochs'],'inner_epoch_selection'))")
    s=replace(s,"        with np.load(dest / 'predictions.npz', allow_pickle=False) as z:","        audit_fit(this,d,c,f,None,dest/'refit',chosen,'outer_refit')\n        require(sha(dest/'refit/latest.pt')==done['checkpoint_sha256'],'Outer checkpoint hash mismatch')\n        with np.load(dest / 'predictions.npz', allow_pickle=False) as z:")
    s=replace(s,"2*len(d['x'][split(d,f,j)[0]].values)","len(d['x'][split(d,f,j)[0]])")
    s=replace(s,"2*len(d['x'][tr].values)","len(d['x'][tr])")
    s=replace(s,"'canonical_fallback_records':int(d['x'].fallback.sum()),","'canonical_fallback_records':int(d['x'].fallback.sum()), 'loss_variant':c['loss_variant'], 'batch_records':c['batch_records'],")
    # Keep old generic smoke out of this runner: verify.py tests actual recording steps/resume.
    a=s.index('def smoke(');b=s.index('def export_review(',a);s=s[:a]+s[b:]
    s=replace(s,"['check', 'smoke', 'pilot', 'train', 'summarize', 'export']","['check', 'pilot', 'train', 'summarize', 'export']")
    s=replace(s,"    args = parser.parse_args()","    parser.add_argument('--variant',choices=['window_control','record_mean'],required=True)\n    args = parser.parse_args()")
    s=replace(s,"    c = config()\n    if args.command == 'smoke':\n        return smoke(c, args.output, args.device)","    c = config();c.update(loss_variant=args.variant,name='cnn_'+args.variant+'_packed_records')")
    s=replace(s,"ROOT / 'windows.py', ROOT / 'input_contract.json'","ROOT / 'windows.py', ROOT / 'record_loss.py', ROOT / 'audit.py', ROOT / 'input_contract.json'")
    (B/'run.py').write_text(s,encoding='utf8')
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
    manifest={p.relative_to(B).as_posix():sha(p) for p in files};save(B/'manifest.json',manifest)
    assert manifest['data.npz']==original['data.npz'] and manifest['model.py']==original['model.py']
    archive=H/'gdi-cnn-record-loss692.zip';prefix='gdi-cnn-record-loss692-20261031/'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,prefix+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(prefix+n)).hexdigest()==h
    save(H/'package_verification.json',dict(passed=True,python38_syntax=True,zip_sha256=sha(archive),manifest_sha256=sha(B/'manifest.json'),data_unchanged=True,model_unchanged=True,server_submitted=False))
    print('Built and hashed standalone pair',flush=True)
if __name__=='__main__':main()
