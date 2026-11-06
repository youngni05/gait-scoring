# PD4T evaluation

The repository retains two complementary analyses: leave-one-subject-out (LOSO) classification from pooled encoder features, and patient-aware clinical-anchor analysis of the six gait estimates.

## Prepare PD4T clips

The released PD4T package is expected to contain `gait_score.xlsx` and either ready-to-use `cropped_videos/` or the following preprocessing inputs:

```text
$PD4T_ROOT/
├── original_videos/<patient-id>/*.mp4
├── PD4T_videos_wham_outputs/<patient-id>/<video-name>/wham_output.pkl
├── wham_pd4t_patient_ids.xlsx
└── gait_score.xlsx
```

If `cropped_videos/` is absent, create it from the provided WHAM tracks:

```bash
python -m PD4T.preprocess_videos --root "$PD4T_ROOT"
```

## LOSO encoder evaluation

Cache one pooled 1024-D feature per cropped clip for GaitXFormer and for the frozen V-JEPA2 baseline:

```bash
python -m PD4T.cache_encoder_out \
  --dataset-path "$PD4T_ROOT/cropped_videos" \
  --checkpoint "$GAITXFORMER_CHECKPOINT_FILE"

python -m PD4T.cache_encoder_out \
  --dataset-path "$PD4T_ROOT/cropped_videos"
```

These commands write `$PD4T_ROOT/cache_gaitxformer` and `$PD4T_ROOT/cache_baseline`. Run the final patient-level LOSO classifiers and compare them:

```bash
python train_pd.py \
  --config configs/pd/default.yaml \
  --log-dir outputs/pd4t/gaitxformer

python train_pd.py \
  --config configs/pd/baseline.yaml \
  --log-dir outputs/pd4t/vjepa2

python pd_report.py \
  --log-dir outputs/pd4t/gaitxformer \
  --log-dir-baseline outputs/pd4t/vjepa2
```

## Patient-aware gait-parameter analysis

Run end-to-end inference followed by the patient-clustered bootstrap analysis:

```bash
python pd4t_patient_aware_clinical_eval.py \
  --config configs/eval/pd4t_patient_aware_clinical_eval.yaml \
  --checkpoint "$GAITXFORMER_CHECKPOINT_FILE" \
  --out-dir outputs/pd4t/patient-aware
```

The output includes raw video predictions, patient/score aggregates, anchor-validity tables, within-patient directional comparisons, and a Markdown summary. To rerun only the statistics from saved predictions, pass `--predictions outputs/pd4t/patient-aware/raw_predictions.csv` instead of `--checkpoint`.
