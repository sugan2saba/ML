# mediwatch/config.py
from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_path(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        node = node.get(k, {})
    return node or default
