import os
import re
from pathlib import Path
import yaml
import json
import random
import torch
import numpy as np
from easydict import EasyDict as edict
from typing import Any, IO


def print_args(args):
    print("[INFO] Input arguments:")
    for key, val in args.items():
        print(f"[INFO]   {key}: {val}")


def create_directory_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def resolve_checkpoint_path(path, default_name="best_ckpt.pth.tr"):
    """Resolve a checkpoint file or a directory containing ``default_name``."""
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / default_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


_ENV_VAR_PATTERN = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def expand_environment_variables(value):
    """Recursively expand environment variables in loaded configuration values."""
    if isinstance(value, dict):
        return {
            key: expand_environment_variables(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment_variables(item) for item in value)
    if not isinstance(value, str):
        return value

    expanded = os.path.expanduser(os.path.expandvars(value))
    unresolved = _ENV_VAR_PATTERN.findall(expanded)
    if unresolved:
        variables = ", ".join(sorted(set(unresolved)))
        raise EnvironmentError(
            f"Configuration references unset environment variable(s): {variables}"
        )
    return expanded


def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(
        loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = expand_environment_variables(config)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config
