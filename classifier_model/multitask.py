"""Run every trained task on the same text, sharing the GPT-2 layers the tasks have in common.

With the default fine-tuning policy only the last block, the final norm and the head
differ between tasks; the embeddings and earlier blocks are the frozen pre-trained weights.
Tasks whose frozen layers are identical form a group: the shared layers ("trunk") run once
per batch and each task runs only its own layers ("branch"). Tasks that cannot share run
as full models ("solo"). Results equal running each bundle on its own (see tests).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
import torch
import yaml

from classifier_model.common_classifier import pool_logits
from classifier_model.bundle import CONFIG_FILE, META_FILE, load_bundle
from classifier_model.config import parse_trainable
from classifier_model.data import PAD_TOKEN_ID, collate, encode_text
from classifier_model.model import select_device

logger = logging.getLogger("classifier_model.multitask")


def frozen_depth(trainable, n_layers):
    """Number of leading transformer blocks left at their pre-trained values."""
    mode, k = parse_trainable(trainable)
    if mode == "head":
        return n_layers
    if mode == "all":
        return 0
    return max(n_layers - k, 0)


class Trunk(torch.nn.Module):
    """The first part of GPTModel.forward: embeddings and the shared blocks."""

    def __init__(self, model, depth):
        super().__init__()
        self.tok_emb = model.tok_emb
        self.pos_emb = model.pos_emb
        self.drop_emb = model.drop_emb
        self.blocks = torch.nn.Sequential(*list(model.trf_blocks)[:depth])

    def forward(self, in_idx):
        positions = torch.arange(in_idx.shape[1], device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(positions))
        return self.blocks(x)


class Branch(torch.nn.Module):
    """The rest of GPTModel.forward: the task's own blocks, final norm and head."""

    def __init__(self, model, depth):
        super().__init__()
        self.blocks = torch.nn.Sequential(*list(model.trf_blocks)[depth:])
        self.final_norm = model.final_norm
        self.out_head = model.out_head

    def forward(self, x):
        return self.out_head(self.final_norm(self.blocks(x)))


@dataclass
class TaskInfo:
    name: str
    description: str | None
    label_names: list
    max_length: int


@dataclass
class Group:
    depth: int
    trunk: Trunk
    branches: dict = field(default_factory=dict)  # task name -> Branch


def _same_trunk(trunk, model, depth):
    ours = trunk.state_dict()
    theirs = Trunk(model, depth).state_dict()
    return ours.keys() == theirs.keys() and all(
        torch.equal(ours[k], theirs[k].to(ours[k].device)) for k in ours
    )


def _read_candidates(artifacts_dir):
    """Read each bundle's small files first, so group depths are known before loading weights."""
    candidates = []
    for meta_path in sorted(Path(artifacts_dir).glob(f"*/{META_FILE}")):
        bundle_dir = meta_path.parent
        if bundle_dir.name.startswith("."):
            continue  # in-progress or leftover temporary bundle
        try:
            meta = json.loads(meta_path.read_text())
            config = yaml.safe_load((bundle_dir / CONFIG_FILE).read_text()) or {}
            model_cfg = meta["model_cfg"]
            depth = frozen_depth(config.get("trainable", {"last_blocks": 1}), model_cfg["n_layers"])
        except Exception as e:  # a bad bundle must not stop the others
            logger.error("Skipping bundle %s: %s", bundle_dir, e)
            continue
        candidates.append((bundle_dir, json.dumps(model_cfg, sort_keys=True), depth))
    return candidates


class MultiTaskClassifier:
    def __init__(self, tasks, groups, solo, device):
        self.tasks = tasks          # task name -> TaskInfo, sorted by name
        self.groups = groups        # list of Group
        self.solo = solo            # task name -> full GPTModel
        self.device = device
        self.max_length = max((t.max_length for t in tasks.values()), default=0)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    @classmethod
    def load(cls, artifacts_dir, device="auto"):
        device = select_device(device)
        candidates = _read_candidates(artifacts_dir)
        group_depth = {}
        for _, key, depth in candidates:
            if depth >= 1:
                group_depth[key] = min(depth, group_depth.get(key, depth))

        tasks, groups_by_key, solo = {}, {}, {}
        for bundle_dir, key, depth in candidates:
            try:
                meta, model = load_bundle(bundle_dir, "cpu")
            except Exception as e:
                logger.error("Skipping bundle %s: %s", bundle_dir, e)
                continue
            name = meta["task"]
            if name in tasks:
                logger.error("Skipping bundle %s: duplicate task name '%s'", bundle_dir, name)
                continue
            tasks[name] = TaskInfo(
                name=name,
                description=meta.get("description"),
                label_names=[label["name"] for label in meta["labels"]],
                max_length=meta["max_length"],
            )
            shared = group_depth.get(key) if depth >= 1 else None
            group = groups_by_key.get(key)
            if shared and group is None:
                group = groups_by_key[key] = Group(depth=shared, trunk=Trunk(model, shared).to(device).eval())
            if shared and _same_trunk(group.trunk, model, shared):
                group.branches[name] = Branch(model, shared).to(device).eval()
            else:
                if shared:
                    logger.warning("Task '%s' has different shared-layer weights; running it as a full model", name)
                solo[name] = model.to(device).eval()

        groups = [g for g in groups_by_key.values() if g.branches]
        for g in groups:
            logger.info("Shared group (blocks 0..%d): tasks %s", g.depth - 1, sorted(g.branches))
        if solo:
            logger.info("Full-model tasks: %s", sorted(solo))
        return cls(dict(sorted(tasks.items())), groups, solo, device)

    def predict(self, texts, batch_size=32):
        """Return, per text in input order, {task: {"label", "confidence", "probabilities"}}."""
        results = []
        for start in range(0, len(texts), batch_size):
            batch = []
            for text in texts[start:start + batch_size]:
                ids = encode_text(text, self.tokenizer, self.max_length)
                batch.append((torch.tensor(ids), torch.tensor(0), torch.tensor(len(ids))))
            input_ids, _, lengths = collate(batch, PAD_TOKEN_ID)
            input_ids = input_ids.to(self.device)
            logits = {}
            with torch.inference_mode():
                for group in self.groups:
                    hidden = group.trunk(input_ids)
                    for name, branch in group.branches.items():
                        logits[name] = self._pool(branch(hidden), lengths, name)
                for name, model in self.solo.items():
                    logits[name] = self._pool(model(input_ids), lengths, name)
                probs = {name: torch.softmax(l.float(), dim=-1).cpu() for name, l in logits.items()}
            for row in range(len(batch)):
                results.append({name: self._result(name, probs[name][row]) for name in self.tasks})
        return results

    def _pool(self, logits, lengths, name):
        # A task with a shorter max_length sees the text truncated to it: with causal
        # attention, position m-1 depends only on tokens 0..m-1.
        task_lengths = torch.clamp(lengths, max=self.tasks[name].max_length).to(logits.device)
        return pool_logits(logits, task_lengths)

    def _result(self, name, row):
        label_names = self.tasks[name].label_names
        best = int(row.argmax())
        return {
            "label": label_names[best],
            "confidence": float(row[best]),
            "probabilities": dict(zip(label_names, row.tolist())),
        }
