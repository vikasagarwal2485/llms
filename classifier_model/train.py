"""Fine-tune GPT-2 into an N-class classifier from a task config.

Usage: python -m classifier_model.train tasks/<task>.yaml [--device auto|cpu|mps|cuda] [--artifacts artifacts]
"""

import argparse
import sys
import time
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

from classifier_model.common_classifier import train_classifier_simple
from classifier_model.bundle import save_bundle
from classifier_model.config import ConfigError, load_config
from classifier_model.data import (
    PAD_TOKEN_ID,
    ClassificationDataset,
    DataError,
    balance_classes,
    build_labels,
    collate,
    encode_labels,
    longest_encoded_length,
    read_split,
)
from classifier_model.metrics import classification_metrics
from classifier_model.model import build_classifier, config_of, predict_logits, select_device


def predict_split(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for input_ids, labels, lengths in loader:
            logits = predict_logits(model, input_ids.to(device), lengths.to(device))
            y_pred.extend(torch.argmax(logits, dim=-1).tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def run_training(cfg, artifacts_dir="artifacts", device="auto", log=print):
    device = select_device(device)
    torch.manual_seed(cfg.seed)

    splits, skipped = {}, {}
    for name in ("train", "val", "test"):
        splits[name], skipped[name] = read_split(getattr(cfg, name), cfg.text_column, cfg.label_column)
        if skipped[name]:
            log(f"Skipped {skipped[name]} row(s) with empty text in the {name} split")
    if cfg.balance:
        splits["train"] = balance_classes(splits["train"], cfg.seed)
    labels = build_labels(splits["train"]["label"], cfg.label_names)
    label_ids = {name: encode_labels(df["label"], labels, name) for name, df in splits.items()}

    model = build_classifier(cfg, num_classes=len(labels))
    context_length = model.pos_emb.weight.shape[0]
    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = min(cfg.max_length or longest_encoded_length(splits["train"]["text"], tokenizer), context_length)

    datasets = {
        name: ClassificationDataset(df["text"], label_ids[name], tokenizer, max_length)
        for name, df in splits.items()
    }
    collate_fn = partial(collate, pad_token_id=PAD_TOKEN_ID)
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
        drop_last=len(datasets["train"]) > cfg.batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    eval_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
        for name, ds in datasets.items()
    }

    log(f"Task '{cfg.task}': {len(labels)} classes {[l['name'] for l in labels]}, "
        f"train/val/test = {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}, "
        f"max_length {max_length}, device {device}")

    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    start = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, eval_loaders["val"], optimizer, device,
        num_epochs=cfg.epochs, eval_freq=cfg.eval_freq, eval_iter=cfg.eval_iter,
    )
    training_seconds = time.time() - start

    label_names = [l["name"] for l in labels]
    metrics = {
        "splits": {
            name: classification_metrics(*predict_split(model, loader, device), label_names)
            for name, loader in eval_loaders.items()
        },
        "history": {
            "train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs, "examples_seen": examples_seen,
        },
        "training_seconds": training_seconds,
        "skipped_rows": skipped,
        "device": str(device),
    }
    for name in ("train", "val", "test"):
        log(f"{name.capitalize()} accuracy: {metrics['splits'][name]['accuracy'] * 100:.2f}%")

    resolved = cfg.to_dict() | {"max_length": max_length}
    model.cpu()
    bundle_path = save_bundle(
        artifacts_dir, cfg.task, model, labels, config_of(model), max_length, PAD_TOKEN_ID, resolved, metrics,
        description=cfg.description,
    )
    log(f"Saved bundle to {bundle_path} (training took {training_seconds / 60:.2f} minutes)")
    return bundle_path, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="Path to the task config YAML")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for model bundles")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        run_training(cfg, args.artifacts, args.device)
    except (ConfigError, DataError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
