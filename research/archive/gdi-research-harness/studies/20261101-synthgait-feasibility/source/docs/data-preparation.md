# Download and cache SynthGait-19K

Training reads preprocessed 32-frame NPZ files rather than decoding and cropping
the released videos during every epoch.

## Download the public dataset

Install the repository requirements, then download the Hugging Face dataset
without manually extracting its WebDataset TAR shards:

```bash
export SYNTHGAIT_ROOT="/path/to/SynthGait-19K"
export SYNTHGAIT_CACHE="/path/to/SynthGait-19K-cache"

hf download SoroushMehraban/SynthGait-19K \
  --repo-type dataset \
  --local-dir "$SYNTHGAIT_ROOT"
```

The downloaded directory contains the dataset card, manifest, checksums, and
approximately 1 GB shards:

```text
$SYNTHGAIT_ROOT/
├── README.md
├── dataset_manifest.json
├── SHA256SUMS
└── data/
    ├── train-00000-of-00101.tar
    ├── ...
    └── validation-00025-of-00026.tar
```

Each walk has one shuffled `vid_XXXXX` base ID. Its available camera views use
that same base:

```text
vid_00001_frontal.mp4
vid_00001_frontal.npz
vid_00001_frontal.json
vid_00001_saggital.mp4
vid_00001_saggital.npz
vid_00001_saggital.json
vid_00001_random.mp4
vid_00001_random.npz
vid_00001_random.json
```

The annotation NPZ contains the six released model targets in this order:

1. `cadence`
2. `walking_speed`
3. `step_length_mean`
4. `step_width_mean`
5. `stoop_posture_mean`
6. `arm_swing`

## Build the training cache

Run the public cache builder from the repository root:

```bash
python cache_hf_dataset.py \
  --dataset-dir "$SYNTHGAIT_ROOT" \
  --output-dir "$SYNTHGAIT_CACHE" \
  --num-workers 8
```

The command reads the shards directly, detects and crops the walker, selects and
pads 32 frames, applies ImageNet normalization, and stores the six named gait
targets. The first run downloads the lightweight YOLO person-detector weights if
they are not already available locally.

Shard names determine the cache split automatically:

- `data/train-*.tar` becomes `$SYNTHGAIT_CACHE/train/`.
- `data/validation-*.tar` becomes `$SYNTHGAIT_CACHE/val/`.

The released split already keeps every camera view of a walk together. Users do
not need original filenames or a separate split manifest.

For GPU-assisted person detection, use one cache worker:

```bash
python cache_hf_dataset.py \
  --dataset-dir "$SYNTHGAIT_ROOT" \
  --output-dir "$SYNTHGAIT_CACHE" \
  --num-workers 1 \
  --detector-device 0
```

## Cache output and disk space

The cache has the layout expected by the training configuration:

```text
$SYNTHGAIT_CACHE/
├── train/
│   ├── vid_00001_frontal.npz
│   ├── ...
│   ├── valid_frame_counts.json
│   └── npz_to_video_map.json
└── val/
    └── ...
```

Each cached NPZ contains `video`, `gait_features`, `feature_names`, and
`frames_valid`. A cached clip is approximately 18 MiB, so allow roughly 350 GiB
for the complete cache in addition to approximately 126 GB for the downloaded
dataset.

Cache creation is restartable. Existing NPZ files are retained unless
`--overwrite` is passed.
