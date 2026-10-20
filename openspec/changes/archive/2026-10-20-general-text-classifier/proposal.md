# Proposal

## Why

The repo has a working GPT-2 classifier, built on the from-scratch `GPTModel` with OpenAI GPT-2 124M open weights and reaching 94.67% test accuracy, but it can only do spam vs. not-spam. The label count, the label names, dataset balancing, the column format and the padding length are all hard-coded for SMS spam. The fine-tuned weights were also never kept: `pre_trained_classifier_gpt_weights_model.pth` is missing, and so is the GPT-2 weight file `model.ckpt.data-00000-of-00001`. As a result nothing in the repo can be run or served today. The aim is a reusable way to fine-tune GPT-2 on *any* labeled text dataset and call the result locally over HTTP.

## What Changes

- **Restore runnable base weights**: fix `gpt_download.py`. Its download call is commented out, and `download_file` is defined twice, with the second definition silently replacing the first, which supports a backup URL. After the fix, GPT-2 124M downloads and loads end to end.
- **General dataset contract**: any CSV with a text column and a label column. Labels can be strings or integers, the number of classes N comes from the training data, optional display names can be given for integer labels, and balancing to the smallest class is optional for any N.
- **Task configuration**: one config file per task (dataset paths, column names, base model size, which layers to unfreeze, epochs, learning rate, batch size, max length, seed). Spam becomes the first task config.
- **N-class classification head** on the existing `GPTModel`: replace `out_head` with `Linear(emb_dim → N)`. Freezing is configurable, and the default matches today's setup (last transformer block, `final_norm` and the head are trainable).
- **Last-real-token pooling**: classify from the final *non-padding* token, not the last padded position. Predictions then stop depending on the padding length, which fixes the mismatch between training at 120 tokens and inference at 1024 in `pre_trained_classifier_gpt_weights_model.py`.
- **Self-describing model bundle**: each trained task saves its weights, label mapping, model and training config, and train/val/test metrics, so it can be loaded without the training code's settings.
- **Training CLI**: train a task from its config and write the bundle and metrics.
- **Prediction API**: load a bundle and classify one text or a batch, returning the label and per-class probabilities.
- **FastAPI service**: serve every bundle found under an artifacts directory. It provides `POST /classify/{task}`, `GET /tasks` and `GET /health`.
- New dependencies: `fastapi`, `uvicorn`, `pyyaml`, `pytest`, `httpx` (for the API test client).

Non-goals: Ollama/GGUF export, Hugging Face `transformers`, generative or instruction-style classification, multi-label classification, and GPU-specific tuning.

## Capabilities

### New Capabilities
- `classifier-training`: fine-tuning GPT-2 into an N-class text classifier from a labeled CSV dataset and a task config, producing a self-describing model bundle with evaluation metrics.
- `classifier-serving`: loading trained model bundles and classifying text, both in-process and through a local HTTP API.

### Modified Capabilities
<!-- None: openspec/specs/ is empty. -->

## Impact

- **Reused unchanged**: `common.py` (`GPTModel`, `load_weights_into_gpt`), the `tiktoken` GPT-2 tokenizer, and the training loop shape from `common_classifier.py` (`train_classifier_simple`, `calc_loss_*`, `calc_accuracy_loader`).
- **Modified**: `gpt_download.py` (restore the download with a backup URL); `common_classifier.py` (generalize the loss and accuracy helpers to pool from the last real token). The spam-specific helpers stay in place for the existing notebooks.
- **New**: a Python package for the general pipeline, a `tasks/` config folder, an `artifacts/` output folder (git-ignored, since weights are about 500 MB), and tests.
- **Existing notebooks**: they are left as they are. `pre_trained_classifier_gpt_weights_model.py` is replaced by the general prediction path, used through the spam task bundle.
- **Disk/network**: about 500 MB to download GPT-2 124M once; about 500 MB per trained task bundle.
