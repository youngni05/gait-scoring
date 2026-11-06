# Downloads and environment configuration

## Install

Create an isolated Python environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your system before the remaining requirements if the default PyPI build does not match your CUDA installation.

## Download the datasets and checkpoint

Download the public SynthGait dataset and GaitXFormer checkpoint. The PD4T URL
remains a placeholder until that release artifact is published:

```bash
export PD4T_DATA_URL="PD4T_DATA_URL_PLACEHOLDER"
export GAITXFORMER_MODEL_DIR="/path/to/GaitXFormer"

hf download SoroushMehraban/SynthGait-19K \
  --repo-type dataset \
  --local-dir /path/to/SynthGait-19K
hf download SoroushMehraban/GaitXFormer gaitxformer.pth.tr \
  --local-dir "$GAITXFORMER_MODEL_DIR"
export GAITXFORMER_CHECKPOINT_FILE="$GAITXFORMER_MODEL_DIR/gaitxformer.pth.tr"
curl -L "$PD4T_DATA_URL" -o PD4T.tar
```

The SynthGait download is already arranged as WebDataset shards; do not unpack
them manually. The checkpoint is downloaded with its released filename,
`gaitxformer.pth.tr`. Extract the PD4T archive after it becomes available; PD4T
access may require accepting the dataset’s own terms once its official URL is
assigned.

## Define paths once

Add these variables to your shell profile, replacing every `/path/to/...` value:

```bash
export SYNTHGAIT_ROOT="/path/to/SynthGait-19K"
export SYNTHGAIT_CACHE="/path/to/SynthGait-19K-cache"
export GAITXFORMER_CHECKPOINT_FILE="/path/to/gaitxformer.pth.tr"
export GAITXFORMER_RUN_DIR="/path/to/GaitXFormer-training-run"
export PD4T_ROOT="/path/to/PD4T"
```

Reload the profile once, for example with `source ~/.bashrc`. The checked-in YAML files use these variables directly, and configuration loading reports an explicit error if a required variable is unset.

`GAITXFORMER_CHECKPOINT_FILE` must name the checkpoint itself, not its containing directory. For example:

```text
$GAITXFORMER_CHECKPOINT_FILE=/models/gaitxformer.pth.tr
```

Build the SynthGait training cache after defining the paths:

```bash
python cache_hf_dataset.py --num-workers 8
```

This command reads the six released gait targets directly from each shard and
uses the included Ultralytics dependency to detect and crop the walker. It does
not require the legacy 15-target source annotations.

The model constructor also uses `torch.hub` to obtain the upstream V-JEPA2 implementation. Internet access is therefore needed on the first run unless it is already present in the local PyTorch Hub cache.

GPJATK is not distributed as a release artifact for this project. See
[the evaluation guide](evaluation.md#gpjatk-availability) for the licensing note.
