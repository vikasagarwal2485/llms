# Design

## Context

See proposal.md for motivation. This is the current state, from `general-text-classifier`:

- `textclf/server.py` builds a `{task: Classifier}` registry at startup and serves `POST /classify/{task}`. Each `Classifier` (`textclf/predict.py`) holds a full GPT-2 copy (about 500 MB).
- A bundle's `config.yaml` records the `trainable` policy, and `bundle.json` records `model_cfg` and `max_length`. Neither records a description.
- `GPTModel.forward` (`common.py`) is `tok_emb + pos_emb -> drop_emb -> trf_blocks -> final_norm -> out_head`. Each piece is a public submodule and can be called on its own.
- `pool_logits` reads the logits at `length - 1`. Attention is causal, so the hidden state at position *i* depends only on tokens `0..i`.
- I measured the `spam` and `ag_news` bundles (default `last_blocks: 1`): 129.9M parameters are identical (embeddings and blocks 0-10) and 7.1M are task-specific (block 11, `final_norm`, `out_head`). `max_length` is 120 for spam and 128 for ag_news.

## Goals / Non-Goals

**Goals:**
- One shared pass per request through the layers that tasks have in common, then a small per-task branch.
- Results identical to per-bundle prediction (spec: Multi-task results match single-task predictions).
- No retraining or format migration for existing bundles.

**Non-Goals:**
- Storing bundles as deltas on disk. Bundles keep their full weights, and sharing happens only in memory.
- Batching requests across HTTP calls, async inference, and GPU-specific optimization.
- Changing the per-bundle `Classifier` and `textclf.predict` CLI. They stay as they are and serve as the reference that tests compare against.

## Decisions

### D1. New module `textclf/multitask.py` with `MultiTaskClassifier`
```
MultiTaskClassifier.load(artifacts_dir, device)
  groups:  [ SharedGroup(trunk, depth D, tasks: {name: Branch}) , ... ]
  solo:    { name: full GPTModel }       # tasks whose layers can't be shared
  predict(texts) -> [ {task: {label, confidence, probabilities}} ... ]
```
`server.py` keeps only HTTP concerns (validation, filtering, status codes) and calls `MultiTaskClassifier`. *Alternative:* put the grouping logic in `server.py`. Rejected because it would make equivalence hard to test without HTTP.

### D2. How many layers each task can share
A task's *frozen depth* `d` comes from its bundle's `config.yaml` `trainable`:

| `trainable` | frozen depth `d` | branch runs |
|---|---|---|
| `head` | `n_layers` | `final_norm`, `out_head` |
| `{last_blocks: K}` | `n_layers - K` | blocks `d..n-1`, `final_norm`, `out_head` |
| `all` | 0 | the whole model (solo) |

Tasks with `d >= 1` are grouped by identical `model_cfg`. The group's shared depth is `D = min(d)` over its members. Tasks that froze more layers than `D` just run a few frozen blocks inside their branch, which is still correct.

### D3. Check the shared weights instead of trusting the config
At load time, each candidate's `tok_emb`, `pos_emb` and blocks `0..D-1` are compared with `torch.equal` against the group's reference trunk. If any tensor differs (for example, a bundle trained from different base weights), that task is moved to `solo` and a warning is logged. The config only *proposes* the sharing depth, and the weights confirm it. *Alternative:* trust `trainable` without checking. Rejected because a wrong guess silently produces wrong answers, and the check is cheap at startup. *Alternative:* hash the tensors. Rejected as unnecessary with only a handful of tasks.

Changing `D` after a group exists (a later task with a smaller `d`) only requires rebuilding the branches. To keep it simple, the loader reads every bundle's config first, computes each group's `D`, and then loads the weights.

### D4. Loading one bundle at a time
Bundles are loaded one at a time. The first bundle in a group provides the trunk modules (`tok_emb`, `pos_emb`, `trf_blocks[:D]`). For every bundle, the branch modules (`trf_blocks[D:]`, `final_norm`, `out_head`) are kept and the rest of the model is released. Peak memory is about two full models during startup. Steady state is one trunk per group plus roughly 28 MB per default-policy task.

### D5. One tokenization and a single forward pass shape
For a request, tokenize each text once, truncated to `L = max(max_length)` over all loaded tasks, and right-pad the batch (existing `collate`). Then:
```
x = drop_emb(tok_emb(ids) + pos_emb(arange(T)))      # same sequence as GPTModel.forward
x = trf_blocks[:D](x)                                 # once per group
for task in group:
    h = out_head(final_norm(branch_blocks(x)))        # [B, T, N_task]
    logits = h[b, min(length_b, max_length_task) - 1]
```
Because attention is causal, the hidden state at position `min(len, m_t) - 1` is exactly what the task's own bundle computes after truncating to `m_t`. Solo tasks run `GPTModel` on the same padded batch and pool the same way. Probabilities come from `softmax` over float32 logits, as in `Classifier.predict`.

### D6. API shape
```
POST /classify
  {"text": str} | {"texts": [str, ...] (1..64)},  "min_confidence": float in [0,1] (optional)
-> 200 {"results": {task: {"label", "confidence", "probabilities": {label: p}}}}          # text
-> 200 {"results": [{task: {...}}, ...]}                                                  # texts
-> 422 invalid body;  503 {"detail": "No tasks are loaded"}
GET /tasks -> {"tasks": [{"task", "description", "labels": [...]}]}
GET /health -> unchanged
```
`POST /classify/{task}` is deleted. With no matching route, FastAPI returns 404. `confidence` is `max(probabilities)`. The `min_confidence` filter is applied in `server.py` after prediction, so `MultiTaskClassifier` always returns every task.

### D7. Description
`TaskConfig.description: str | None = None` is validated as a string when present. `save_bundle` writes `"description"` into `bundle.json`, and loaders read it with `meta.get("description")`, so older bundles yield `None`. `format_version` stays 1, because the field is optional and additive.

## Risks / Trade-offs

- [Every task answers every text, including off-domain ones (the spam model judging a news headline)] → `description` in `/tasks` lets the agent judge relevance, and `min_confidence` drops weak answers. Automatic routing stays a non-goal.
- [Latency grows with the number of tasks] → Each extra default-policy task adds one transformer block plus a head per request, about 1/12 of a GPT-2 pass. Solo tasks add a full pass, and startup logs which tasks run solo.
- [Small float differences from different padded shapes (the multi-task batch pads to the longest `L`)] → Masked positions are exactly zero after softmax. Equivalence is tested on CPU to within 1e-5.
- [Breaking API removal] → The only in-repo clients are the tests and the README, and both are updated. The migration note is in the spec.
- [Startup peak memory with many bundles] → Sequential loading (D4) keeps the peak at about two full models.

## Migration Plan

1. Deploy the new server. Existing `artifacts/*` bundles load unchanged and are listed with `description: null`.
2. Optionally add `description` to `tasks/*.yaml` and retrain to fill it in.
3. Rollback: revert the code changes and remove `description` from `tasks/*.yaml`, because the older config loader rejects unknown fields. Bundles with a `description` in `bundle.json` still load under the old code, which ignores the extra key.
