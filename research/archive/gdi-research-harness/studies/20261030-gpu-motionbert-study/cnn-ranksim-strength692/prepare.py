"""Prepare a fresh, narrowly changed RankSim-strength round; never alter old runs."""
from pathlib import Path
import json,hashlib,shutil,ast
H=Path(__file__).resolve().parent;OLD=H.parent/'cnn-ranksim692'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 p=H/name
 if not isinstance(value,str):value=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
 p.write_text(value,encoding='utf-8')
assert not (H/'submission.json').exists(),'Already submitted; never rebuild active bundle'
for n in ['run.py','model.py','windows.py','balanced.py','audit.py','review.py','review_results.py','verify.py','worker.py','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE','reference_metrics.json','input_contract.json']:
 shutil.copyfile(OLD/n,H/n)
c=read(OLD/'protocol.json');c.update(name='cnn_ranksim_strength',stage='ranksim_strength',note='Only rank-loss strength changes; reuse verified exact MSE0/RankSim1 controls. Choose strength/epoch on inner people only.',variants=['rank01','rank03'],rank_weights={'rank01':.1,'rank03':.3},rank_weight=.1,primary_strength_grid=[0,.1,.3,1],expected_fits=50,expected_outer_folds=10)
c.update(hypothesis='Weaker rank regularization may retain complementary ranking information while reducing point-regression cost',same_cohort_people=692,same_records=1061,same_windows=11261,selection_over_strength='minimum of each fixed-strength mean inner calibrated MAE curve; ties1e-12 prefer smaller strength then earlier epoch',reuse_control_remote='/abr/RESEARCH_USER/gdi-cnn-ranksim692-20261101',reuse_control_manifest_sha256=sha(OLD/'manifest.json'),source_result_digest=sha(OLD/'analysis/comparison.json'),source_ensemble_digest=sha(H.parent/'ranksim-ensemble692/analysis/comparison.json'),primary_metric='person_equal_MAE',secondary_metrics=['person_mean90/100 MAE and bias','observation90/100 MAE and bias','low-score error','same10/20band accuracy','5fold changes','paired person bootstrap10000 seed20261101'],promotion='overall delta<0,95%upper<0,4/5fold improvement; high90 MAE and absolute bias not worse for joint promotion',expected_GPU_hours_total=[7,12],expected_GPU_memory_GiB_below=1,pilot_gate='all3device synthetic checks + both3epoch GDI pilots + source GPU/CPU64 + local CPU64 and hashes, before any full training',protected_final_or_LJMU=False)
write('protocol.json',c)
s=(H/'run.py').read_text(encoding='utf-8').replace("c['stage'] == 'person_balanced_mse_bmc_fixed_sigma'","c['stage'] == 'ranksim_strength'")
needle="def person_mae(y, pred, people):"
fn="""def variant_config(variant):
    c=config()
    require(variant in ['mse','rank01','rank03'],'Invalid strength arm')
    c.update(loss_variant='mse' if variant=='mse' else 'ranksim', rank_weight=0. if variant=='mse' else c['rank_weights'][variant], rank_variant=variant, name='cnn_'+variant+'_person_balanced')
    return c


"""
s=s.replace(needle,fn+needle)
s=s.replace("choices=['mse','ranksim']","choices=['rank01','rank03']")
s=s.replace("c = config();c.update(loss_variant=args.variant,name='cnn_'+args.variant+'_person_balanced')","c = variant_config(args.variant)")
write('run.py',s)
s=(H/'review.py').read_text(encoding='utf-8').replace("choices=['mse','ranksim']","choices=['rank01','rank03']")
s=s.replace("c=r.config();c.update(loss_variant=a.variant,name='cnn_'+a.variant+'_person_balanced')","c=r.variant_config(a.variant)")
write('review.py',s)
s=(H/'review_results.py').read_text(encoding='utf-8').replace("['mse','ranksim']","['rank01','rank03']")
s=s.replace("c=r.config();c.update(loss_variant=variant,name='cnn_'+variant+'_person_balanced')","c=r.variant_config(variant)")
# The old report's paired keys are auxiliary labels; actual primary strength
# selection is handled by select_strength.py from inner curves only.
s=s.replace("errors['ranksim']-errors['mse']","errors['rank03']-errors['rank01']").replace("ranksim_minus_mse_mae=","rank03_minus_rank01_mae=")
write('review_results.py',s)
s=(H/'verify.py').read_text(encoding='utf-8')
s=s.replace("for variant in ['mse','ranksim']:","for variant in ['mse','rank01','rank03']:")
s=s.replace("cc=dict(c,loss_variant=variant,name='synthetic_'+variant)","cc=dict(c,loss_variant='mse' if variant=='mse' else 'ranksim',rank_weight=0. if variant=='mse' else c['rank_weights'][variant],name='synthetic_'+variant)")
s=s.replace("cc=dict(c,max_epochs=1,loss_variant=variant,name='synthetic_nested_'+variant)","cc=dict(c,max_epochs=1,loss_variant='mse' if variant=='mse' else 'ranksim',rank_weight=0. if variant=='mse' else c['rank_weights'][variant],name='synthetic_nested_'+variant)")
s=s.replace("synthetic_nested_fits=50,synthetic_outer_folds=10","synthetic_nested_fits=75,synthetic_outer_folds=15")
s=s.replace("env=r.setup(42,a.device);c=r.config();d=r.data()","env=r.setup(42,a.device);c=r.config();d=r.data()\n    from controls import verify_reuse\n    verify_reuse(r)")
s=s.replace("# Independent log-sum-exp expression", "from select_strength import choose\n    assert choose([dict(strength=.3,epoch=1,score=2.),dict(strength=0.,epoch=2,score=2.+5e-13)])['strength']==0.\n    assert choose([dict(strength=.1,epoch=3,score=1.),dict(strength=.3,epoch=1,score=.8)])['strength']==.3\n    # Independent log-sum-exp expression")
write('verify.py',s)
s=(H/'worker.py').read_text(encoding='utf-8').replace("['mse','ranksim']","['rank01','rank03']")
s=s.replace("variant=['mse','ranksim'][rank]","variant=['rank01','rank03'][rank]")
needle="        info['phase']='main';r.save(state,info)"
assert s.count(needle)==1
s=s.replace(needle,"""        info['phase']='waiting_local_pilot_audit';r.save(state,info)
        wait_for(lambda:(control/'approve-main-local.json').exists(),seconds=10800)
        local=r.load(control/'approve-main-local.json')
        assert local['passed'] and local['manifest_sha256']==gate['manifest_sha256'] and local['proofs']==gate['proofs']
        assert local['local_report_sha256']
"""+needle)
s=s.replace("r.save(root/'runs/pair_summary.json',dict(variants=summaries,fits=50,outer_folds=10))","execute(['select_strength.py'])\n            r.save(root/'runs/pair_summary.json',dict(variants=summaries,fits=50,outer_folds=10,strength_selection_sha256=r.sha(root/'runs/strength_selection.json')))")
s=s.replace("'PERSON_BMC_WORKER_FINISHED'","'RANK_STRENGTH_WORKER_FINISHED'")
write('worker.py',s)
# Prove no changes to training computation, sampling, epoch selection or model.
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old=functions(OLD/'run.py');new=functions(H/'run.py')
critical=['fit','step','predict','make_optimizer','split','scale_from_train','calibrate','corrected','select_epoch','train_fold','identity_for']
assert all(old[k]==new[k] for k in critical)
write('source-provenance.json',dict(original_manifest_sha256=sha(OLD/'manifest.json'),unchanged_functions=critical,unchanged_files={n:sha(H/n) for n in ['model.py','windows.py','balanced.py','rank_aux.py','ranking.py','ranksim.py','RANKSIM_LICENSE']},reference_metrics_included=True,only_training_change='rank_weight0.1 or0.3',control_result_sha256=sha(OLD/'analysis/comparison.json')))
for p in H.glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
print(json.dumps(dict(prepared=True,unchanged_training_functions=critical)))
if __name__=='__main__':pass
