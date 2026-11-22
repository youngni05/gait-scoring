"""Prepare the pinned MediaPipe and GDI files in a directory outside the repo."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


POSE_MODEL = "pose_landmarker_full.task"
POSE_MODEL_SHA256 = "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"
POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/"
    + POSE_MODEL + "?generation=1682642787774579"
)
GDI_PACKAGES = {
    "baseline": ("gdi-frozen54-20261106", "f545ad9501658ebf6d15030180b2a8f658a1d8472c6b09b564748c7b71813eb4"),
    "normal100": ("gdi-normal100-paper75-20261106", "b0cb2393dafdfa01665d6e2f41c1a9618d252aa4d7c5b9aad5b3a2689c7bafda"),
}
RELEASE = "https://github.com/youngni05/gait-scoring/releases/download/research-snapshot-20261106/"
STANFORD = "https://raw.githubusercontent.com/stanfordnmbl/mobile-gaitlab/4167ee6213ac81fb9168f4e20c6538d42a41a5a6/"
STANFORD_FILES = {
    "training/video_process_utils.py": "0dab14d35b2bc3eaabad8901c7d02a9ec08fa2bb8d3077a2866b57579a6ae1ac",
    "training/process_raw_videos.ipynb": "34188bf66081858b2395b173375b87f510543a5d8d6ff7093bcd52299d8b1b1e",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def download(path, url, expected):
    if path.is_file():
        verify(path, expected)
        return
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify(temporary, expected)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(assets):
    assets.mkdir(parents=True, exist_ok=True)
    download(assets / POSE_MODEL, POSE_URL, POSE_MODEL_SHA256)
    for package, checksum in GDI_PACKAGES.values():
        archive = package + ".tar.gz"
        download(assets / archive, RELEASE + archive, checksum)
        with tarfile.open(assets / archive, "r:gz") as bundle:
            bundle.extractall(assets, filter="data")
    sources = assets / "stanford"
    sources.mkdir(exist_ok=True)
    for relative, checksum in STANFORD_FILES.items():
        download(sources / Path(relative).name, STANFORD + relative, checksum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.assets.resolve())
    print(f"Verified analysis assets: {args.assets.resolve()}")
