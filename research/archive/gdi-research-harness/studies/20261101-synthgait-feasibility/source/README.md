# SynthGait-19K

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2609.08108)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-87CEFA)](https://huggingface.co/datasets/SoroushMehraban/SynthGait-19K)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-model-FFD21E)](https://huggingface.co/SoroushMehraban/GaitXFormer)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97-space-FFC266)](https://huggingface.co/spaces/SoroushMehraban/GaitXFormer)

This is the official repository for **“SynthGait-19K: A Physically Grounded Synthetic Video Dataset for Gait Parameter Estimation.”** It contains GaitXFormer, its training and validation pipelines, the PD4T evaluation pipeline, and the retained GPJATK evaluation code.

## Documentation

| Guide | Description |
| --- | --- |
| [Downloads and environment](docs/downloads-and-environment.md) | Download and configure the datasets and released checkpoint. |
| [Data preparation](docs/data-preparation.md) | Preprocess SynthGait-19K and build the training cache. |
| [Training](docs/training.md) | Train or resume the final GaitXFormer configuration. |
| [Validation](docs/evaluation.md) | Validate the public SynthGait split and review the GPJATK availability note. |
| [PD4T](docs/pd4t.md) | Run LOSO classification and patient-aware clinical analysis. |

## Download and cache SynthGait-19K

The public dataset is released as Hugging Face WebDataset shards. Download the
snapshot and build the 32-frame training cache directly from those shards:

```bash
export SYNTHGAIT_ROOT="/path/to/SynthGait-19K"
export SYNTHGAIT_CACHE="/path/to/SynthGait-19K-cache"

hf download SoroushMehraban/SynthGait-19K \
  --repo-type dataset \
  --local-dir "$SYNTHGAIT_ROOT"

python cache_hf_dataset.py \
  --dataset-dir "$SYNTHGAIT_ROOT" \
  --output-dir "$SYNTHGAIT_CACHE" \
  --num-workers 8
```

The downloaded shard names already encode the fixed train/validation split, so
clients do not need the original filenames or a separate split step. The cache
builder reads the six released gait targets directly and detects/crops the walker
from each RGB video. See
[data preparation](docs/data-preparation.md) for the file schema, preprocessing
details, and disk requirements.

## Video demo

The video demo needs the repository and released checkpoint only; it does not require SynthGait-19K, GPJATK, or PD4T. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

Download the released checkpoint from the
[GaitXFormer model repository](https://huggingface.co/SoroushMehraban/GaitXFormer):

```bash
export GAITXFORMER_MODEL_DIR="/path/to/GaitXFormer"

hf download SoroushMehraban/GaitXFormer gaitxformer.pth.tr \
  --local-dir "$GAITXFORMER_MODEL_DIR"

export GAITXFORMER_CHECKPOINT_FILE="$GAITXFORMER_MODEL_DIR/gaitxformer.pth.tr"
```

Then run:

```bash
python demo.py --video samples/sample1.mp4
```

`demo.py` selects the middle five seconds, samples at 16 FPS, detects and crops the walker, runs the 32-frame GaitXFormer model, and prints the six denormalized gait estimates in a terminal table. Input videos should show one visible, full-body walker. Videos shorter than five seconds use their full duration. The first run downloads the lightweight YOLO person detector and the V-JEPA2 model definition through their upstream packages.

Mean Stooped Posture and Arm Swing are dimensionless ratios normalized by leg length, so the demo does not show units for them.

Pass a checkpoint directly or optionally save a machine-readable JSON result:

```bash
python demo.py \
  --video /path/to/walking_video.mp4 \
  --checkpoint /path/to/gaitxformer.pth.tr \
  --json-output prediction.json
```

To launch the local Gradio interface using the same `GAITXFORMER_CHECKPOINT_FILE`:

```bash
python gradio/app.py
```
