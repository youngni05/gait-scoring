import random

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

import utils.misc as misc
from data.cached_gait_est import CachedGaitEstimationDataset
from model.gaitxformer import GaitXFormer


class AverageMeter:
    """Compute and store an average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(args, opts):
    """Build the cached real-data loaders used by the final training run."""
    train_dataset = CachedGaitEstimationDataset(**args.data, split="train")
    val_dataset = CachedGaitEstimationDataset(**args.data, split="val")

    if opts.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        validation_sampler = DistributedSampler(
            val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        validation_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train.batch_size,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=validation_sampler,
        batch_size=1,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
    )
    return train_dataloader, val_dataloader


def load_model(args, opts, device):
    model = GaitXFormer(**args.model, n_out=len(args.data.gait_features))
    model.to(device)

    model_without_ddp = model
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opts.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module
    return model, model_without_ddp


def save_model(
    model_without_ddp,
    optimizer,
    min_error,
    epoch,
    wandb_id,
    checkpoint_path,
):
    """Preserve the parameter-only checkpoint format used by the final run."""
    if not misc.is_main_process():
        return

    learnable_names = {
        name
        for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad
    }
    learnable_state_dict = {
        name: value
        for name, value in model_without_ddp.state_dict().items()
        if name in learnable_names
    }
    if "best" in checkpoint_path:
        payload = {"model": learnable_state_dict}
    else:
        payload = {
            "model": learnable_state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "min_error": min_error,
            "wandb_id": wandb_id,
        }
    torch.save(payload, checkpoint_path)
