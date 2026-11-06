from glob import glob
import argparse
import os
import joblib
import pandas as pd
import ast
import cv2
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
from data.utils.affine import generate_image_patch_cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess videos for PD4T")
    parser.add_argument("--root", type=str, required=True, help="Directory containing input data")
    parser.add_argument("--patient-id", type=str, default=None, help="Optional patient id to process, e.g. 066")
    parser.add_argument("--debug", action="store_true", help="Write debug overlay video and exit")
    return parser.parse_args()


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_trimmed_verts(track, frame_start, frame_end):
    verts_all = _to_numpy(track["verts"])
    while verts_all.ndim > 3 and verts_all.shape[0] == 1:
        verts_all = verts_all[0]
    if verts_all.ndim == 2 and verts_all.shape[-1] == 3:
        verts_all = verts_all[None, ...]
    if verts_all.ndim != 3 or verts_all.shape[-1] != 3:
        raise ValueError(f"Unexpected verts shape: {verts_all.shape}")

    frame_ids = track.get("frame_ids", None)

    if frame_ids is None:
        rel_start = max(0, int(frame_start))
        rel_end = min(int(frame_end), int(verts_all.shape[0] - 1))
        if rel_end < rel_start:
            return np.empty((0,) + verts_all.shape[1:], dtype=verts_all.dtype), np.empty((0,), dtype=np.int64)
        trimmed_verts = verts_all[rel_start: rel_end + 1]
        trimmed_frame_ids = np.arange(rel_start, rel_end + 1, dtype=np.int64)
        return trimmed_verts, trimmed_frame_ids

    frame_ids = _to_numpy(frame_ids)
    frame_ids = np.asarray(frame_ids).reshape(-1).astype(np.int64)

    if frame_ids.shape[0] != verts_all.shape[0]:
        min_len = min(frame_ids.shape[0], verts_all.shape[0])
        frame_ids = frame_ids[:min_len]
        verts_all = verts_all[:min_len]

    keep = (frame_ids >= int(frame_start)) & (frame_ids <= int(frame_end))
    trimmed_verts = verts_all[keep]
    trimmed_frame_ids = frame_ids[keep]
    return trimmed_verts, trimmed_frame_ids


def _project_and_draw_verts(frame, verts_3d, focal_length=5000.0):
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    verts = _to_numpy(verts_3d)
    if verts.ndim != 2 or verts.shape[-1] != 3:
        return frame

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    # WHAM outputs can be in either +Z-forward or -Z-forward convention.
    if np.mean(z > 0.0) >= 0.5:
        z_cam = z
    else:
        z_cam = -z

    valid = z_cam > 1e-6
    if not np.any(valid):
        return frame

    # Match WHAM camera model: K = [[f,0,cx],[0,f,cy],[0,0,1]], R=I, T=0.
    # Keep image-space y-down convention used by WHAM demo rendering.
    f = float(focal_length)
    u = (f * (x[valid] / z_cam[valid])) + cx
    v = (f * (y[valid] / z_cam[valid])) + cy

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    for px, py in zip(u[in_bounds], v[in_bounds]):
        cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 0), -1)

    return frame


def _project_verts_to_pixels(verts_3d, width, height, focal_length=5000.0):
    verts = _to_numpy(verts_3d)
    if verts.ndim != 2 or verts.shape[-1] != 3:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    cx, cy = width / 2.0, height / 2.0
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    if np.mean(z > 0.0) >= 0.5:
        z_cam = z
    else:
        z_cam = -z

    valid = z_cam > 1e-6
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    f = float(focal_length)
    u = (f * (x[valid] / z_cam[valid])) + cx
    v = (f * (y[valid] / z_cam[valid])) + cy

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_bounds], v[in_bounds]


def _get_projected_bbox_params(verts_3d, width, height, focal_length, margin=10.0):
    u, v = _project_verts_to_pixels(verts_3d, width, height, focal_length=focal_length)
    if u.size == 0 or v.size == 0:
        return None

    min_x, max_x = float(np.min(u)), float(np.max(u))
    min_y, max_y = float(np.min(v)), float(np.max(v))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    bbox_width = (max_x - min_x) + float(margin) * 2.0
    bbox_height = (max_y - min_y) + float(margin) * 2.0
    bbox_size = max(bbox_width, bbox_height)

    if bbox_size <= 1.0:
        return None

    return center_x, center_y, bbox_size


def _write_debug_video(video_path, track, frame_start, frame_end, out_path="debug.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video for debug: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    verts_trimmed, frame_ids_trimmed = _get_trimmed_verts(track, frame_start, frame_end)
    if len(frame_ids_trimmed) == 0:
        print(f"No verts in frame range [{frame_start}, {frame_end}] for debug output")
        cap.release()
        return

    frame_to_vert_idx = {int(fid): idx for idx, fid in enumerate(frame_ids_trimmed.tolist())}
    writer = imageio.get_writer(
        out_path,
        fps=float(fps),
        codec="libx264",
        macro_block_size=None,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
    first_overlay_done = False
    for frame_idx in range(int(frame_start), int(frame_end) + 1):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in frame_to_vert_idx:
            v_idx = frame_to_vert_idx[frame_idx]
            h, w = frame.shape[:2]
            focal_length = (w ** 2 + h ** 2) ** 0.5
            frame = _project_and_draw_verts(frame, verts_trimmed[v_idx], focal_length=focal_length)
            if not first_overlay_done:
                first_overlay_done = True
                print(
                    f"Overlaying verts on frame {frame_idx} | "
                    f"verts shape per-frame: {verts_trimmed[v_idx].shape}"
                )

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    writer.close()
    cap.release()
    print(f"Saved debug video: {out_path}")


def _write_cropped_video(video_path, track, frame_start, frame_end, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video for cropping: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            print(f"Could not read first frame for cropping: {video_path}")
            return
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))

    verts_trimmed, frame_ids_trimmed = _get_trimmed_verts(track, frame_start, frame_end)
    if len(frame_ids_trimmed) == 0:
        print(f"No verts in frame range [{frame_start}, {frame_end}] for cropping")
        cap.release()
        return

    focal_length = (width ** 2 + height ** 2) ** 0.5
    frame_to_vert_idx = {int(fid): idx for idx, fid in enumerate(frame_ids_trimmed.tolist())}

    per_frame_params = []
    bbox_sizes = []
    for frame_idx in range(int(frame_start), int(frame_end) + 1):
        v_idx = frame_to_vert_idx.get(frame_idx, None)
        if v_idx is None:
            per_frame_params.append(None)
            continue
        params = _get_projected_bbox_params(
            verts_3d=verts_trimmed[v_idx],
            width=width,
            height=height,
            focal_length=focal_length,
            margin=10.0,
        )
        per_frame_params.append(params)
        if params is not None:
            bbox_sizes.append(float(params[2]))

    if len(bbox_sizes) == 0:
        print(f"Could not compute valid projected bbox for [{frame_start}, {frame_end}]")
        cap.release()
        return

    # Fill missing frames by nearest available params.
    last_valid = None
    for i in range(len(per_frame_params)):
        if per_frame_params[i] is not None:
            last_valid = per_frame_params[i]
        elif last_valid is not None:
            per_frame_params[i] = last_valid

    next_valid = None
    for i in range(len(per_frame_params) - 1, -1, -1):
        if per_frame_params[i] is not None:
            next_valid = per_frame_params[i]
        elif next_valid is not None:
            per_frame_params[i] = next_valid

    if any(x is None for x in per_frame_params):
        print(f"Insufficient bbox params for [{frame_start}, {frame_end}]")
        cap.release()
        return

    target_size = int(np.ceil(max(bbox_sizes)))
    target_size = max(2, target_size)
    if target_size % 2 == 1:
        target_size += 1

    # Match downstream crop behavior:
    # offset = self.height // 8
    # frame = frame[:, offset:-offset]
    offset = target_size // 8
    final_width = target_size - (2 * offset)
    if final_width < 2:
        offset = 0
        final_width = target_size
    if final_width % 2 == 1:
        if offset > 0:
            offset -= 1
            final_width = target_size - (2 * offset)
        else:
            final_width -= 1

    writer = imageio.get_writer(
        out_path,
        fps=float(fps),
        codec="libx264",
        macro_block_size=None,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
    for i, _ in enumerate(range(int(frame_start), int(frame_end) + 1)):
        ok, frame = cap.read()
        if not ok:
            break

        center_x, center_y, bbox_size = per_frame_params[i]
        crop, _ = generate_image_patch_cv2(
            frame,
            float(center_x),
            float(center_y),
            float(bbox_size),
            float(bbox_size),
            int(target_size),
            int(target_size),
            False,
            1.0,
            0.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        if offset > 0:
            crop = crop[:, offset:-offset]

        if crop.shape[1] != final_width:
            crop = crop[:, :final_width]

        writer.append_data(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    writer.close()
    cap.release()
    print(f"Saved cropped video: {out_path}")

def main():
    args = parse_args()
    video_paths = glob(f"{args.root}/original_videos/*/*.mp4")
    if args.patient_id is not None:
        patient_id = str(args.patient_id).zfill(3)
        video_paths = [
            p for p in video_paths
            if os.path.basename(os.path.dirname(p)) == patient_id
        ]

    df = pd.read_excel(os.path.join(args.root, 'wham_pd4t_patient_ids.xlsx'))
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        patient_id = video_path.split("/")[-2]
        video_name = video_path.split("/")[-1]

        walk_name = f"{patient_id}-{video_name.split('.')[0]}"
        try:
            row = df[df["Walk Name"] == walk_name].iloc[0]
        except Exception as e:
            continue
        try:
            wham_ids = [int(x.strip()) for x in str(row["WHAM IDs"]).split(",")]
        except Exception as e:
            continue
        frames = ast.literal_eval(row["Frames"])

        wham_path = os.path.join(args.root, "PD4T_videos_wham_outputs", patient_id, video_name, 'wham_output.pkl')
        with open(wham_path, 'rb') as f:
            wham_output = joblib.load(f)

        out_dir = os.path.join(args.root, "cropped_videos")
        os.makedirs(out_dir, exist_ok=True)
        
        if len(wham_ids) == 1:
            frames = [frames]

        segment_idx = 0
        
        for wham_id, valid_frames in zip(wham_ids, frames):
            track = wham_output[wham_id]
            for frame_start, frame_end in valid_frames:
                if args.debug:
                    print(video_path, wham_id, frame_start, frame_end)
                    _write_debug_video(
                        video_path=video_path,
                        track=track,
                        frame_start=int(frame_start),
                        frame_end=int(frame_end),
                        out_path="debug.mp4",
                    )
                    return
                else:
                    out_path = os.path.join(out_dir, f"{walk_name}_{segment_idx}.mp4")
                    if os.path.exists(out_path):
                        print(f"Output already exists, skipping: {out_path}")
                        segment_idx += 1
                        continue
                    _write_cropped_video(
                        video_path=video_path,
                        track=track,
                        frame_start=int(frame_start),
                        frame_end=int(frame_end),
                        out_path=out_path,
                    )
                    segment_idx += 1


if __name__ == "__main__":
    main()

"""Run with: python -m PD4T.preprocess_videos --root "$PD4T_ROOT"."""
