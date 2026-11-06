"""Cache pooled PD4T V-JEPA2 features for the final model and baseline."""

import argparse
import os
import torch
from glob import glob
from tqdm import tqdm
from decord import VideoReader, cpu
from model.gaitxformer import GaitXFormer
from utils.tools import resolve_checkpoint_path
from data.utils import volume_transforms
import numpy as np
import data.utils.transforms as video_transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    pd4t_root = os.environ.get('PD4T_ROOT')
    default_dataset = os.path.join(pd4t_root, 'cropped_videos') if pd4t_root else None
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=default_dataset,
        required=default_dataset is None,
        help='Directory of cropped PD4T videos (defaults to $PD4T_ROOT/cropped_videos)',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        metavar='PATH',
        help='Final checkpoint file or directory; omit for the frozen V-JEPA2 baseline',
    )
    parser.add_argument('--patient-id', type=str, default=None, help='Optional patient id to cache, e.g. 066')
    opts = parser.parse_args()
    return opts


def load_backbone(checkpoint, device):
    model = GaitXFormer(
        encoder_name="vjepa2_vit_large",
        n_decoder_layers=3,
        expansion_ratio=2,
        patch_t=2,
        patch_h=16,
        patch_w=16,
        decoder_type='cross_attn',
        encoder_freeze_strategy='full_finetune' if checkpoint is not None else 'freeze_encoder',
        camera_tokens=False,
        n_out=6
    )
    model.to(device)

    if checkpoint is not None:
        checkpoint_path = resolve_checkpoint_path(checkpoint)
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
        state_dict = checkpoint_payload.get('model', checkpoint_payload)
        model.load_state_dict(state_dict, strict=True)
        print(f'[INFO] GaitXFormer initialized from {checkpoint_path}.')
    else:
        print('[INFO] Baseline (V-JEPA2) initialized.')
    return model.encoder


def subsample_frames(frames, max_frames=81, n_frames=32):
    """
    subsample based on max_frames. If frames < max_frames, subsamples and returns less than self.num_frames and will be zero-padded later.
    Args:
        frames: Input frames with shape (T, H, W) or (T, H, W, C)
    Returns:
        Subsampled frames with shape (<=n_frames, H, W) or (<=n_frames, H, W, C)
    """
    frame_indices = np.linspace(0, max_frames - 1, n_frames, dtype=int)

    T = frames.shape[0]
    
    if T > max_frames:
        start_idx = (T - max_frames) // 2
        end_idx = start_idx + max_frames
        frames = frames[start_idx:end_idx]
    
    frame_indices = frame_indices[frame_indices < frames.shape[0]]
    return frames[frame_indices]

def zero_pad_frames(video_frames, n_frames=32):
    T = video_frames.shape[0]
    assert T <= n_frames, "Number of frames exceeds n_frames"
    valid_mask = np.zeros((n_frames,), dtype=bool)
    valid_mask[:T] = True
    
    if T == n_frames:
        return video_frames, valid_mask
    
    # Calculate padding needed
    pad_frames = n_frames - T
    
    # Zero pad video frames
    video_pad_shape = (pad_frames,) + video_frames.shape[1:]
    video_padding = np.zeros(video_pad_shape, dtype=video_frames.dtype)
    video_frames_padded = np.concatenate([video_frames, video_padding], axis=0)
    
    return video_frames_padded, valid_mask


def resample_frames(n_frames, original_fps, target_fps):
    if original_fps !=target_fps:
        duration = n_frames / original_fps
        target_frames = int(duration *target_fps)
        
        assert target_frames > 0, "Target frames after resampling is zero."
        
        frame_indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
    else:
        frame_indices = np.arange(n_frames)
    return frame_indices


def load_video(video_path, transforms, target_fps=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    original_fps = vr.get_avg_fps()
    n_frames = len(vr)
    frame_indices = resample_frames(n_frames, original_fps, target_fps)
    frames = vr.get_batch(frame_indices).asnumpy()

    frames = subsample_frames(frames, max_frames=81, n_frames=32)
    frames, valid_mask = zero_pad_frames(frames, n_frames=32)
    frames = transforms(frames)
    valid_mask = torch.from_numpy(valid_mask)
    return frames, valid_mask

@torch.no_grad()
def main():
    opts = parse_args()

    videos = sorted(glob(os.path.join(opts.dataset_path, "*.mp4")))
    if opts.patient_id is not None:
        patient_id = str(opts.patient_id).zfill(3)
        videos = [
            video_path for video_path in videos
            if os.path.basename(video_path).startswith(f"{patient_id}-")
        ]

    dataset_root_path = os.path.dirname(os.path.normpath(opts.dataset_path))
    
    cache_name = f'cache_{"baseline" if opts.checkpoint is None else "gaitxformer"}'
    cache_path = os.path.join(dataset_root_path, cache_name)
    
    os.makedirs(cache_path, exist_ok=True)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_backbone(opts.checkpoint, device)

    transforms = video_transforms.Compose(
            [
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )


    for video_path in tqdm(videos):
        video_name = os.path.basename(video_path)
        out_path = os.path.join(cache_path, f"{os.path.splitext(video_name)[0]}.pt")
        if os.path.exists(out_path):
            print(f"Cache already exists, skipping: {out_path}")
            continue
        
        video, valid_mask = load_video(video_path, transforms)
        video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)
        
        valid_f = valid_mask.to(device).unsqueeze(0)  # (1, T)

        B, _, T, H, W = video.shape
        features = backbone(video)

        valid_f = valid_f[:, ::2].float()                     # (B, T//2)
        valid_f = valid_f.view(-1, T//2, 1, 1, 1)            # (B, T//2, 1, 1, 1)
        features = features.reshape(B, T//2, H//16, W//16, -1)
        masked_feats = features * valid_f          # zero out invalid frames

        lengths = valid_f.sum(dim=1)               # (B, 1, 1, 1), number of valid frames
        lengths = lengths.clamp(min=1.0)           # avoid div-by-zero

        feats = masked_feats.sum(dim=1) / lengths  # (B, H/16, W/16, encoder_dim)
        feats = feats.mean(dim=(1, 2))             # spatial average pooling -> (B, encoder_dim)

        torch.save(feats.cpu(), out_path)

if __name__ == "__main__":
    main()


"""
For V-JEPA2:
 python3 -m PD4T.cache_encoder_out

For GaitXFormer:
 python3 -m PD4T.cache_encoder_out --checkpoint "$GAITXFORMER_CHECKPOINT_FILE"
"""
