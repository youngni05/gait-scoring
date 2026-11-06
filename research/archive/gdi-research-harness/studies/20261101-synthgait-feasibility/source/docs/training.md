# Training GaitXFormer

Create the cache first using [the data-preparation guide](data-preparation.md). The final configuration is `configs/gait_estimation/32f_crossattn_fullyfinetune.yaml`: 32 frames, six gait targets, an MSE objective, a three-layer cross-attention decoder, and full V-JEPA2 fine-tuning.

For one GPU:

```bash
python train_gait_est.py \
  --config configs/gait_estimation/32f_crossattn_fullyfinetune.yaml \
  --checkpoint "$GAITXFORMER_RUN_DIR"
```

For distributed training on four local GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train_gait_est.py \
  --config configs/gait_estimation/32f_crossattn_fullyfinetune.yaml \
  --checkpoint "$GAITXFORMER_RUN_DIR"
```

Add `--use-wandb --wandb-name NAME` to record the run in Weights & Biases. The trainer writes `best_ckpt.pth.tr` and the resumable `last_ckpt.pth.tr` inside the output directory. Re-running the same command resumes automatically when `last_ckpt.pth.tr` exists; use a new empty directory to train from scratch.

The batch size in the released configuration is per process. Adjust `train.batch_size` and the learning rate deliberately if the number of GPUs or available memory differs from the original run.
