# SynthGait validation

The commands below use the exact checkpoint file specified by `GAITXFORMER_CHECKPOINT_FILE`. They write per-video predictions, correlation plots, text summaries, and `metrics.json` to `--corr-dir`.

The validation evaluator reads the cached `val/` split created from the Hugging
Face shards. Set `SYNTHGAIT_CACHE` to the same cache root used for training:

```bash
python eval_gait_est.py \
  --config configs/eval/validation.yaml \
  --checkpoint "$GAITXFORMER_CHECKPOINT_FILE" \
  --view all \
  --corr-dir outputs/validation-all
```

Use `--view preferred` for the feature-specific preferred-view protocol or `--view avg` to compute each metric by view and combine the view correlations with a Fisher z-transform.

The evaluator derives each camera view from cached sample names such as
`vid_00001_frontal.npz` and reports the corresponding `.mp4` sample name. It
does not require raw videos, segmentation files, original filenames, or a
separate split manifest.

## GPJATK availability

The GPJATK evaluation code is retained in this repository for transparency, but
the GPJATK data and the derived per-sequence gait annotations cannot be released
with this project. The [official GPJATK dataset page](https://zenodo.org/records/15130800)
states that the dataset is available only for scientific use and that its release
agreement must be accepted; the agreement restricts redistribution without prior
written approval from the dataset owner. For that reason, this public guide does
not provide a GPJATK data or ground truth gait parameters derived from it.
