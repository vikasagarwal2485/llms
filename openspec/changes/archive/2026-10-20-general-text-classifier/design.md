# Design

## Context

See proposal.md (Why) for motivation. This is the current state that shapes the approach:

- `common.py` holds the from-scratch `GPTModel` (Raschka, *Build a Large Language Model From Scratch*) and `load_weights_into_gpt`, which maps OpenAI's TF checkpoint arrays onto the PyTorch modules. No change is needed there.
- `gpt_download.py` loads GPT-2 through TensorFlow (`tf.train.load_variable`). Its `download_file(...)` call is commented out, failures are only printed rather than raised, and an interrupted download can leave a partial file. (The second, `requests`-based `download_file` sits inside a string literal and has no effect.) `gpt2/124M/` is missing `model.ckpt.data-00000-of-00001`.
- `common_classifier.py` has a reusable training loop (`train_classifier_simple`, `evaluate_model`, `calc_loss_*`, `calc_accuracy_loader`), but it pools with `model(x)[:, -1, :]`, which reads the last *padded* position. It also has spam-only helpers (`SpamDataset`, `create_balanced_dataset`, `classify_review`).
- The notebook trains with `max_length = 120` (the longest training message), batch size 8, AdamW at lr 5e-5 and weight decay 0.1, 5 epochs, seed 123, and only the last block, `final_norm` and the head unfrozen. That run reached 94.67% test accuracy.
- The `.myenv` virtual environment has torch 2.14, tiktoken, tensorflow 2.21, pandas, numpy, tqdm and requests. It does not have fastapi, uvicorn, pyyaml or pytest.
- The spam CSVs use the columns `Label,Text` with integer labels (0 = ham, 1 = spam).

## Goals / Non-Goals

**Goals:**
- Reuse `GPTModel`, `load_weights_into_gpt`, `download_and_load_gpt2` and the training loop instead of rewriting them.
- Keep the existing notebooks working: changes to `common_classifier.py` must stay backward compatible.
- Serving must not need TensorFlow or the GPT-2 base checkpoint, only a trained bundle.

**Non-Goals:**
- Hugging Face `transformers`, ONNX, GGUF or Ollama.
- Multi-label classification, sample weighting, early stopping and hyperparameter search.
- Authentication, rate limiting and deployment beyond localhost.
- Training from a single unsplit CSV. Datasets arrive as train, validation and test files.

## Decisions

### D1. New package `textclf/` beside the existing modules
```
textclf/
  config.py     task config (YAML) -> validated dataclass, defaults
  data.py       ClassificationDataset, label mapping, balancing, collate
  model.py      build_classifier(): GPTModel + base weights + N-class head + freeze policy
  train.py      CLI: python -m textclf.train tasks/<task>.yaml
  metrics.py    confusion matrix, per-class P/R/F1 (numpy, no sklearn)
  bundle.py     save/load bundle (atomic write)
  predict.py    Classifier.load(path).predict(texts); CLI: python -m textclf.predict
  server.py     FastAPI app factory; python -m textclf.server
tasks/spam.yaml
artifacts/<task>/      (git-ignored)
tests/
```
`textclf` imports `GPTModel` from `common` and the generalized loop from `common_classifier`, and is run from the repo root (`python -m ...`). *Alternative:* add it all to `common_classifier.py`. Rejected because that mixes notebook-era spam code with the pipeline and makes serving import TensorFlow-dependent modules.

### D2. Pool at the last real token, passing lengths through the existing loop
Datasets yield `(input_ids, label, length)`. A collate function right-pads each batch with `<|endoftext|>` (50256) to that batch's longest sequence, capped at `max_length`. Because GPT-2 attention is causal, a token never sees the padding after it, so the logits at index `length - 1` are exactly what the unpadded text would produce.

`common_classifier` gains `pool_logits` and `unpack_batch`. If a batch has 3 elements, pooling uses `logits[arange(B), lengths - 1]`, otherwise `[:, -1, :]` as today. `calc_loss_batch` takes an optional `lengths`, and the loops in `calc_loss_loader`, `calc_accuracy_loader` and `train_classifier_simple` unpack batches through `unpack_batch` (they previously unpacked exactly two elements). `evaluate_model` is unchanged. The notebooks' 2-tuple batches behave exactly as before.

*Alternative:* left padding. Rejected because GPT-2's absolute positional embeddings would shift every token's position. *Alternative:* keep fixed padding with last-position pooling. Rejected because predictions would depend on padding length, which is the bug in `pre_trained_classifier_gpt_weights_model.py`.

Per-batch padding also speeds up training and inference compared with padding every example to the dataset maximum.

### D3. Freeze policy
The config field `trainable` accepts `head`, `last_blocks: K` (default K=1, which also unfreezes `final_norm`) or `all`. This follows the notebook: freeze everything, replace `out_head` with `Linear(emb_dim, N)` (trainable by default), then unfreeze the selected modules. The optimizer only receives parameters with `requires_grad`.

### D4. Labels
The classes are the sorted unique training labels, compared as strings so that `0`/`1` and `ham`/`spam` are handled the same way. The mapping `id -> {value, name}` is saved in the bundle. `label_names` in the config gives display names, which matters for integer labels. A validation or test label that isn't in the training set is a hard error (spec: General labeled dataset).

### D5. Bundle format
```
artifacts/<task>/
  model.pth      full state_dict (incl. N-class head), loaded with weights_only=True
  bundle.json    {format_version, task, labels[], model_cfg (BASE_CONFIG), max_length, pad_token_id, tokenizer: "gpt2"}
  config.yaml    resolved task config (defaults filled in)
  metrics.json   per-split accuracy, per-class P/R/F1/support, confusion matrix, loss history, train time
```
The bundle is written to `artifacts/.<task>.tmp-<pid>/` and then `os.replace`d into place, so a failed run never leaves a half-written bundle. *Alternative:* save only the trainable parameters (about 7M parameters, roughly 28 MB, for the default policy) and rebuild from the base checkpoint at load time. Rejected for v1 because serving would then need TensorFlow and the base weights. It's a possible later size optimization.

`max_length` defaults to `min(longest tokenized training text, context_length)`, matching the notebook, and is stored in the bundle so inference truncates the same way.

### D6. Base weights download
Fix `gpt_download.py`: restore the call, skip files that already exist, download to a `.part` file and rename it when complete, raise an error naming the file when both URLs fail, and use `certifi`'s CA bundle, because python.org macOS builds ship without root certificates. `download_and_load_gpt2` and `load_weights_into_gpt` then run unchanged, and TensorFlow is imported only on the training path (`textclf.model`). Serving and prediction import only `common.GPTModel`, `torch` and `tiktoken`.

### D7. Serving
`create_app(artifacts_dir, device)` scans `artifacts/*/bundle.json` at startup and loads each bundle into a `{task: Classifier}` registry. A bundle that fails to load is logged and skipped. Endpoints are `GET /health`, `GET /tasks` and `POST /classify/{task}`. The request body is a pydantic model that accepts exactly one of `text` or `texts`, with non-empty strings and at most 64 texts. Inference runs under `torch.inference_mode()`, and batch prediction reuses the D2 collate path. `python -m textclf.server --artifacts artifacts --host 127.0.0.1 --port 8000` starts uvicorn.

### D8. Device and reproducibility
Device order is `cuda`, then `mps`, then `cpu`, with a `--device` override. Seeding sets `torch.manual_seed` and gives the training DataLoader its own seeded `Generator`. `drop_rate` stays 0.0. Exact reproducibility is guaranteed on CPU only (spec: Reproducible training).

### D9. Dependencies
Add a `requirements.txt` that pins the packages already installed (torch, tiktoken, tensorflow, pandas, numpy, tqdm, requests, certifi) plus the new ones (fastapi, uvicorn, pydantic, pyyaml, pytest, httpx).

## Risks / Trade-offs

- [Spam accuracy changes when pooling moves from the last padded token to the last real token] → The spam parity requirement (>= 94%) is the acceptance gate. If the default settings fall short, adjust epochs or `last_blocks` in `tasks/spam.yaml` only, not the defaults.
- [Each bundle is about 500 MB on disk and in memory, so several tasks in one server multiply RAM] → This is acceptable for local use. Saving only the trainable parameters (D5 alternative) is the documented way to shrink it.
- [TensorFlow is a heavy training-only dependency] → It's already installed and used by existing code. It's confined to the training path, so serving never imports it.
- [The GPT-2 download mirrors may change or be unreachable] → There's a primary URL plus the backup mirror, files that already exist are skipped, and the error names the failing file.
- [MPS results can differ slightly from CPU] → Reproducibility is only claimed on CPU, and parity is checked on whichever device trained.
- [Unbalanced datasets can bias the classifier toward the majority class] → `balance: true` is available, and per-class metrics make the bias visible.

## Migration Plan

1. The notebooks and `common_classifier.py` callers keep working, because the 2-tuple batch behavior is unchanged.
2. `pre_trained_classifier_gpt_weights_model.py` becomes a thin wrapper that loads `artifacts/spam` through `textclf.predict`. Its spam-specific behavior (with 1024-token padding) goes away.
3. Rollback: delete `textclf/`, `tasks/` and `artifacts/`, then revert `gpt_download.py` and `common_classifier.py`.
