"""One CPU tracker per anonymous video. Only coordinates go to stdout."""
import ctypes
import hashlib
import json
import os
import resource
import sys
from importlib.metadata import version


def extract(video, model):
    import av
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    source_hash = hashlib.file_digest(video, "sha256").hexdigest()
    video.seek(0)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
    )
    frames = []
    previous_ms = -1
    geometry = None
    with av.open(video, format="mov", options={"enable_drefs": "0", "use_absolute_path": "0",
                                               "protocol_whitelist": "pipe"}) as container:
        stream, = container.streams.video
        if stream.width * stream.height > 4096 * 2160:
            raise ValueError("image size")
        stream.codec_context.thread_count = 1
        with vision.PoseLandmarker.create_from_options(options) as tracker:
            for frame in container.decode(stream):
                if len(frames) >= 2000 or frame.pts is None or frame.width * frame.height > 4096 * 2160:
                    raise ValueError("frame limits")
                seconds = float(frame.pts * frame.time_base)
                ms = round(seconds * 1000)
                if ms <= previous_ms or seconds < 0:
                    raise ValueError("frame clock")
                previous_ms = ms
                rotation = frame.rotation
                if rotation % 90:
                    raise ValueError("unsupported display matrix")
                for side in frame.side_data:
                    if side.type.name == "DISPLAYMATRIX":
                        matrix = np.frombuffer(side, dtype=np.int32).reshape(3, 3)
                        if np.linalg.det(matrix[:2, :2].astype(float)) <= 0:
                            raise ValueError("mirrored video")
                rgb = np.rot90(frame.to_ndarray(format="rgb24"), rotation // 90)
                height, width = rgb.shape[:2]
                if geometry is not None and geometry != (width, height, rotation):
                    raise ValueError("changing geometry")
                geometry = width, height, rotation
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                marks = tracker.detect_for_video(image, ms).pose_landmarks
                frames.append({"t": seconds, "lm": [[v.x, v.y, v.z, v.visibility] for v in marks[0]] if marks else None})
        if not frames or (stream.frames and stream.frames != len(frames)):
            raise ValueError("incomplete video")
    width, height, rotation = geometry
    return dict(width=width, height=height, rotation=rotation, source_sha256=source_hash,
                engine_version=version("mediapipe"), frames=frames)


if __name__ == "__main__":
    os.umask(0o077)
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    # End the tracker even if the API process dies while holding the request.
    ctypes.CDLL(None).prctl(1, 9, 0, 0, 0)
    if os.getppid() != int(sys.argv[2]):
        sys.exit(1)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (130, 135))
    with os.fdopen(0, "rb", closefd=False) as video:
        json.dump(extract(video, sys.argv[1]), sys.stdout, separators=(",", ":"), allow_nan=False)
