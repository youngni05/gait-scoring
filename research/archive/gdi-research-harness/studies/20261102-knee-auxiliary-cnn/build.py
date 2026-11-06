"""Prepare a standalone sealed experiment; never start real optimization."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil
import numpy as np
from audit_inputs import audited_data, sha, read, BASE, TABLE

HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE / 'submission.json').exists(), 'Do not reseal submitted code'
    data, aux, _ = audited_data()
    for source, dest in [('run.py', 'baseline.py'), ('model.py', 'model.py'), ('windows.py', 'windows.py')]:
        assert sha(BASE / source) == read(BASE / 'manifest.json')[source]
        shutil.copyfile(BASE / source, HERE / dest)
    shutil.copyfile(BASE / 'sources/LICENSE', HERE / 'SOURCE_LICENSE')
    shutil.copyfile(BASE / 'sources/provenance.json', HERE / 'SOURCE_PROVENANCE.json')
    np.savez_compressed(HERE / 'auxiliary-targets.npz', aux=aux[:, :, 1],
                        **{k: data[k] for k in ['person_id', 'video_id', 'fold']})
    # Verify that the extracted clinical CSV exactly matches the archived original.
    archive_tool = HERE.parents[2] / 'tools/stanford-sample-check/inspect_archive.py'
    spec = importlib.util.spec_from_file_location('archive_source', archive_tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = read(TABLE.parent / 'archive-index.json')
    member = next(m for m in index['non_json_members'] if m['name'].endswith('/alldata.csv'))
    assert hashlib.sha256(module.read_member(member)).hexdigest() == sha(TABLE)
    c = read(BASE / 'protocol.json')
    c.update(name='matched_GDI_CNN_with_clinical_knee_auxiliary', stage='knee_auxiliary_nested_comparison',
             auxiliary_target='KneeFlex_maxExtension', auxiliary_weights=[0.1, 0.3, 1.0],
             auxiliary_scaling='independent_train_only_pooled_sides_minmax',
             auxiliary_output='existing_unused_output_index_1_no_architecture_change',
             auxiliary_loss='same_record_window_weights_MSE_on_output1',
             total_loss='original_GDI_MSE_plus_original_kernel_L2_plus_lambda_knee_MSE',
             selection='lowest_inner_person_weighted_GDI_calibrated_MAE; ties smaller_lambda_then_earlier_epoch',
             expected_fits=90, expected_outer_predictions=10,
             fit_counts={'gdi_only':25, 'knee_auxiliary':65},
             data_sha256=sha(BASE / 'data.npz'), auxiliary_sha256=sha(HERE / 'auxiliary-targets.npz'),
             baseline_manifest_sha256=sha(BASE / 'manifest.json'), clinical_table_sha256=sha(TABLE),
             primary_comparison='knee_auxiliary_minus_same_condition_gdi_only',
             uncertainty={'draws':10000, 'seed':20261102, 'unit':'person', 'paired':True, 'confidence':0.95},
             decision='CI_upper_below_zero_improved; CI_lower_above_zero_worse; otherwise_inconclusive',
             reserved_holdout_used=False, independent_final_evaluation=False,
             authorization='2026-11-02 user approved target, three inner-selected weights, 90 fits, school GPU',
             inference_inputs='video features only; true knee targets never required at inference',
             notes=['Repeated development-data exploration; no new confirmatory holdout claim.',
                    'Record-balanced original CNN loss and original initialization retained; no ensemble.',
                    'Original second output is reused; same parameter count and random initialization.',
                    'External paper predicts knee separately; this experiment tests joint learning.'])
    (HERE / 'protocol.json').write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files = ['baseline.py', 'model.py', 'windows.py', 'run.py', 'verify.py', 'review.py', 'worker.py',
             'protocol.json', 'auxiliary-targets.npz', 'SOURCE_LICENSE', 'SOURCE_PROVENANCE.json']
    for name in files:
        if name.endswith('.py'):
            ast.parse((HERE / name).read_text(encoding='utf-8'), feature_version=(3,8))
    manifest = {name:sha(HERE / name) for name in files}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(manifest_sha256=sha(HERE / 'manifest.json'), original_data_sha256=c['data_sha256'],
                          auxiliary_clinical_csv_matches_original_archive=True, real_optimizer_steps=0)))


if __name__ == '__main__':
    main()
