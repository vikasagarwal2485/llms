"""Self-describing model bundles: weights, labels, inference settings, config, metrics."""

import json
import os
import shutil
from pathlib import Path

import torch
import yaml

from textclf.model import create_classifier

FORMAT_VERSION = 1
WEIGHTS_FILE = "model.pth"
META_FILE = "bundle.json"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.json"
REQUIRED_FILES = (WEIGHTS_FILE, META_FILE, CONFIG_FILE, METRICS_FILE)


class BundleError(ValueError):
    pass


def save_bundle(artifacts_dir, task, model, labels, model_cfg, max_length, pad_token_id, config, metrics,
                description=None):
    """Write the bundle to a temporary directory, then swap it into artifacts_dir/task."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_dir = artifacts_dir / task
    tmp_dir = artifacts_dir / f".{task}.tmp-{os.getpid()}"
    old_dir = artifacts_dir / f".{task}.old-{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()

    torch.save(model.state_dict(), tmp_dir / WEIGHTS_FILE)
    meta = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "description": description,
        "labels": labels,
        "model_cfg": model_cfg,
        "max_length": max_length,
        "pad_token_id": pad_token_id,
        "tokenizer": "gpt2",
    }
    (tmp_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    (tmp_dir / CONFIG_FILE).write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))

    if final_dir.exists():
        os.replace(final_dir, old_dir)
    os.replace(tmp_dir, final_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
    return final_dir


def load_bundle(path, device="cpu"):
    """Return (meta, model in eval mode on device)."""
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"Bundle directory does not exist: {path}")
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            raise BundleError(f"Bundle {path} is missing {name}")
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"Bundle {path} has unsupported format_version {meta.get('format_version')!r}")
    model = create_classifier(meta["model_cfg"], len(meta["labels"]))
    state_dict = torch.load(path / WEIGHTS_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return meta, model.to(device).eval()


def load_metrics(path):
    return json.loads((Path(path) / METRICS_FILE).read_text())
