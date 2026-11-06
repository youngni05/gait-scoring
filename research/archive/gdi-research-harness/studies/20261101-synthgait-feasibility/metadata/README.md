---
pretty_name: SynthGait-19K
license: other
license_name: synthgait-19k-research-use-license
license_link: LICENSE
annotations_creators:
- machine-generated
source_datasets:
- original
size_categories:
- 10K<n<100K
tags:
- video
- synthetic
- gait-analysis
- biomechanics
- webdataset
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.tar
  - split: validation
    path: data/validation-*.tar
---

# SynthGait-19K

<div align="center">
  <video autoplay loop muted controls playsinline style="max-width: 100%;">
    <source src="https://huggingface.co/datasets/SoroushMehraban/SynthGait-19K/resolve/main/teaser.mp4" type="video/mp4">
  </video>
</div>

SynthGait-19K contains 19,272 synthetic walking videos from
6,427 underlying motion sequences. Each motion is rendered from up to
three camera viewpoints. The dataset supports video-based gait parameter estimation.

## Dataset structure

The release uses uncompressed WebDataset TAR shards of approximately 1 GB. The shared
`vid_XXXXX` base identifies one walking sequence and the suffix identifies its camera
view. Each full stem forms one WebDataset sample:

```text
vid_00001_frontal.mp4   # H.264 video, 832x480, 16 FPS
vid_00001_frontal.npz   # sanitized six-target numeric annotation
vid_00001_frontal.json  # viewer-friendly metadata
vid_00001_saggital.mp4
vid_00001_saggital.npz
vid_00001_saggital.json
vid_00001_random.mp4
vid_00001_random.npz
vid_00001_random.json
```

| Split | Videos | Walks |
| --- | ---: | ---: |
| train | 15,397 | 5,135 |
| validation | 3,875 | 1,292 |

Related views are assigned to the same split. View totals are frontal
6,425, sagittal 6,426, and random
6,421.

The JSON member contains `sample_id`, `split`, `view`, video properties, camera rotation
and field of view, named gait features, and left/right heel-strike indices. The NPZ
contains the released `features`, `left_heel_strikes`, and `right_heel_strikes` arrays.

Heel-strike indices refer to the **30 Hz annotation timeline**, not directly to the 16 FPS
video frame indices.

## Gait feature schema

The `features` NPZ vector and the `gait_features` JSON object use this order:
`cadence, walking_speed, step_length_mean, step_width_mean, stoop_posture_mean, arm_swing`. These are the six output targets in the released
GaitXFormer configuration. All other legacy source features are excluded from both
formats.

| Field | Unit | Meaning |
| --- | --- | --- |
| `cadence` | steps/min | Cadence |
| `walking_speed` | m/s | Mean forward walking speed |
| `step_length_mean` | m | Mean step length |
| `step_width_mean` | m | Mean step width |
| `stoop_posture_mean` | ratio | Mean C7-to-S1 forward offset divided by leg length |
| `arm_swing` | ratio | Wrist forward-range divided by leg length |

## Download and build the training cache

The repository's cache builder consumes these shards directly. After cloning the
FoundationGait repository and installing its requirements:

```bash
export SYNTHGAIT_ROOT="/path/to/SynthGait-19K"
export SYNTHGAIT_CACHE="/path/to/SynthGait-19K-cache"

hf download YOUR_NAMESPACE/SynthGait-19K \
  --repo-type dataset \
  --local-dir "$SYNTHGAIT_ROOT"

python cache_hf_dataset.py \
  --dataset-dir "$SYNTHGAIT_ROOT" \
  --output-dir "$SYNTHGAIT_CACHE" \
  --num-workers 8
```

The shard name determines the split: `train-*.tar` becomes cache split `train`, and
`validation-*.tar` becomes cache split `val`. The split is finalized during packaging,
with every view of an underlying walk kept in the same split; clients do not need a
separate filename manifest.

`cache_hf_dataset.py` reads the six released targets by name and uses a person
detector to construct the cropped 32-frame tensors expected by the training loader. It
does not require the excluded legacy gait targets.

## Public identifiers and metadata

Walks are shuffled before public `vid_XXXXX` identifiers are assigned. Camera views of
the same motion share that base identifier, so no separate `walk_id` is needed. The
release does not include original filenames, subject identifiers, source-dataset labels,
or the private `dataset_mapping.json` file. Train versus validation membership is
intentionally visible in the shard path and JSON `split` field.

These measures remove direct source identifiers from the release, but they should be
described as pseudonymization rather than a guarantee of irreversible anonymity. An
organization that already possesses the source motions could potentially match records
using motion or annotation similarity.

## Other loading options

With Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("YOUR_NAMESPACE/SynthGait-19K", streaming=True)
sample = next(iter(dataset["train"]))
```

With WebDataset:

```python
import webdataset as wds

url = "https://huggingface.co/datasets/YOUR_NAMESPACE/SynthGait-19K/resolve/main/data/train-{00000..00100}-of-00101.tar"
dataset = wds.WebDataset(url, shardshuffle=True).decode().to_tuple("mp4", "npz", "json")
```

## License

SynthGait-19K is available under the
[SynthGait-19K Research Use License](LICENSE). It may be used for non-commercial
research only. Commercial, production, clinical, diagnostic, and patient-care uses
are prohibited without prior written permission from the SynthGait-19K authors.

Synthetic data can reproduce biases or artifacts from its motion sources and generation
pipeline. This dataset is not clinically validated and is not a substitute for clinical
data or professional medical judgment.
