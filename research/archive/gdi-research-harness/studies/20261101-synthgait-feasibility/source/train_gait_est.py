"""Train the final GaitXFormer model.

Final launch command::

    python train_gait_est.py \
      --config configs/gait_estimation/32f_crossattn_fullyfinetune.yaml \
      --checkpoint "$GAITXFORMER_RUN_DIR" \
      --use-wandb --wandb-name 32frames_mse_rank64_crossattn_fullyfinetune

Passing the checkpoint directory again resumes from ``last_ckpt.pth.tr``.
An explicit resumable checkpoint file can be passed instead.
"""

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch import optim
from tqdm import tqdm

import utils.misc as misc
from utils.learning import AverageMeter, load_data, load_model, save_model
from utils.tools import (
    create_directory_if_not_exists,
    get_config,
    print_args,
    set_random_seed,
)


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-sd", "--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--config",
        default="configs/gait_estimation/32f_crossattn_fullyfinetune.yaml",
        help="Path to the final training config",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        metavar="PATH",
        default="checkpoint",
        help=(
            "Checkpoint output directory, or an explicit last checkpoint file "
            "from which to resume"
        ),
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-name", default=None)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser.parse_args()


def checkpoint_paths(path):
    """Return output directory, resume file, best file, and last file."""
    expanded_path = os.path.abspath(os.path.expanduser(path))
    filename = os.path.basename(expanded_path)
    is_explicit_file = os.path.isfile(expanded_path) or filename.endswith(
        (".pt", ".pth", ".pth.tr", ".ckpt")
    )
    if is_explicit_file:
        checkpoint_dir = os.path.dirname(expanded_path)
        resume_path = expanded_path
    else:
        checkpoint_dir = expanded_path
        resume_path = os.path.join(checkpoint_dir, "last_ckpt.pth.tr")
    return (
        checkpoint_dir,
        resume_path,
        os.path.join(checkpoint_dir, "best_ckpt.pth.tr"),
        os.path.join(checkpoint_dir, "last_ckpt.pth.tr"),
    )


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for videos, valid, targets in tqdm(data_loader):
        videos = videos.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(videos, valid)
        if outputs.shape != targets.shape:
            raise RuntimeError(
                f"Model output shape {tuple(outputs.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        loss = torch.mean((outputs - targets) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), videos.size(0))

    torch.cuda.empty_cache()
    return {"train/loss_gait": loss_meter.avg, "train/loss": loss_meter.avg}


@torch.no_grad()
def evaluate(args, model, data_loader, device):
    model.eval()
    num_features = len(args.data.gait_features)
    sum_gait_errors = torch.zeros(num_features, device=device)
    sum_norm_gait_error = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    all_outputs = []
    all_targets = []

    for videos, valids, targets in tqdm(data_loader):
        videos = videos.to(device, non_blocking=True)
        valids = valids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(videos, valids)
        unnormalized_outputs = data_loader.dataset.denormalize_gait_features(outputs)
        unnormalized_targets = data_loader.dataset.denormalize_gait_features(targets)

        all_outputs.append(unnormalized_outputs.cpu())
        all_targets.append(unnormalized_targets.cpu())
        gait_errors = torch.abs(unnormalized_outputs - unnormalized_targets)
        normalized_gait_errors = torch.abs(outputs - targets)
        batch_size = videos.size(0)
        sum_gait_errors += gait_errors.mean(dim=0) * batch_size
        sum_norm_gait_error += normalized_gait_errors.mean() * batch_size
        count += batch_size

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_gait_errors)
        torch.distributed.all_reduce(sum_norm_gait_error)
        torch.distributed.all_reduce(count)

    result = {
        f"eval/error_{feature_name}": error
        for feature_name, error in zip(
            args.data.gait_features,
            (sum_gait_errors / count).detach().cpu().tolist(),
        )
    }
    result["eval/error_all"] = (sum_norm_gait_error / count).item()

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    if torch.distributed.is_initialized():
        all_outputs = concat_all_gather(all_outputs.to(device)).cpu()
        all_targets = concat_all_gather(all_targets.to(device)).cpu()

    outputs_array = all_outputs.numpy()
    targets_array = all_targets.numpy()
    correlations = {
        f"eval/corr_{feature_name}": np.corrcoef(
            outputs_array[:, index],
            targets_array[:, index],
        )[0, 1]
        for index, feature_name in enumerate(args.data.gait_features)
    }
    correlations["eval/corr_all"] = np.mean(list(correlations.values()))
    result.update(correlations)
    return result


def load_resume_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "epoch", "wandb_id"}
    missing_fields = required.difference(checkpoint)
    if missing_fields:
        raise ValueError(
            f"Training resume checkpoint {path} is missing fields: "
            f"{sorted(missing_fields)}. Use last_ckpt.pth.tr rather than the "
            "parameter-only best checkpoint."
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint.get("min_error", float("inf")),
        checkpoint["wandb_id"],
    )


def train(args, opts):
    print_args(args)
    if str(getattr(args.train, "loss", "mse")).lower() != "mse":
        raise ValueError("The retained final trainer supports only train.loss='mse'.")

    checkpoint_dir, resume_path, best_path, last_path = checkpoint_paths(
        opts.checkpoint
    )
    if misc.is_main_process():
        create_directory_if_not_exists(checkpoint_dir)

    if torch.cuda.is_available() and opts.distributed:
        device = torch.device(f"cuda:{opts.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader = load_data(args, opts)
    model, model_without_ddp = load_model(args, opts, device)
    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model_without_ddp.parameters()),
        lr=args.train.lr,
    )

    resume = os.path.isfile(resume_path)
    if resume:
        print(f"[INFO] Resuming from {resume_path}")
        start_epoch, min_error, wandb_id = load_resume_checkpoint(
            resume_path,
            model_without_ddp,
            optimizer,
        )
    else:
        start_epoch = 0
        min_error = float("inf")
        wandb_id = wandb.util.generate_id()

    if opts.use_wandb and misc.is_main_process():
        if resume:
            wandb.init(
                id=wandb_id,
                project="GaitXFormer",
                resume="must",
                settings=wandb.Settings(start_method="fork"),
            )
        else:
            wandb.init(
                name=opts.wandb_name,
                project="GaitXFormer",
                settings=wandb.Settings(start_method="fork"),
            )
            wandb.config.update({"run_id": wandb_id})
            wandb.config.update(args)
            wandb_id = wandb.run.id

    print(f"[INFO] Starting from epoch {start_epoch}")
    for epoch in range(start_epoch, args.train.epochs):
        print(f"[INFO] Epoch {epoch}")
        if opts.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_result = train_one_epoch(model, train_dataloader, optimizer, device)
        torch.cuda.empty_cache()
        eval_result = evaluate(args, model, val_dataloader, device)
        torch.cuda.empty_cache()

        if eval_result["eval/error_all"] < min_error:
            min_error = eval_result["eval/error_all"]
            save_model(
                model_without_ddp,
                optimizer,
                min_error,
                epoch,
                wandb_id,
                best_path,
            )
        if misc.is_main_process():
            log_dict = {
                **train_result,
                **eval_result,
                "eval/min_error_all": min_error,
            }
            if opts.use_wandb:
                wandb.log(log_dict, step=epoch + 1)
            for key, value in log_dict.items():
                print(f"[INFO] {key}: {value}")

        save_model(
            model_without_ddp,
            optimizer,
            min_error,
            epoch,
            wandb_id,
            last_path,
        )


def main():
    opts = parse_args()
    misc.init_distributed_mode(opts)
    set_random_seed(opts.seed + misc.get_rank())
    cudnn.benchmark = True
    args = get_config(opts.config)

    if not opts.distributed:
        global concat_all_gather
        concat_all_gather = lambda tensor: tensor

    train(args, opts)


if __name__ == "__main__":
    main()
