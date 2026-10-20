# Tasks

## 1. Setup and base weights

- [x] 1.1 Add `requirements.txt` (D9), install fastapi, uvicorn, pydantic, pyyaml, pytest and httpx into `.myenv`, and verify `python -c "import fastapi, uvicorn, yaml, pytest, httpx"` succeeds
- [x] 1.2 Fix `gpt_download.py` (D6): restore the download call, raise when both URLs fail, write via a `.part` file, use certifi CAs, with backup-URL fallback, and skip files that already exist. Verify that `download_and_load_gpt2("124M", "gpt2")` downloads `model.ckpt.data-00000-of-00001` and returns params whose `wte` shape is (50257, 768)
- [x] 1.3 Add `artifacts/` to `.gitignore` and verify `git check-ignore artifacts/x` matches
- [x] 1.4 Create the `textclf/` package and `tests/` folder (D1) and verify `python -c "import textclf"` succeeds from the repo root

## 2. Reusable loop generalization

- [x] 2.1 Generalize `calc_loss_batch` and `calc_accuracy_loader` in `common_classifier.py` to accept `(x, y, lengths)` batches and pool at `lengths - 1`, keeping the `(x, y)` behavior unchanged (D2). Verify with a unit test that a 2-tuple batch gives the same loss as before and that a 3-tuple batch pools at the given index
- [x] 2.2 Confirm `train_classifier_simple` and `evaluate_model` run unchanged with 3-tuple batches by running a 2-step smoke training on a tiny random `GPTModel` in a test

## 3. Config and data

- [x] 3.1 Implement `textclf/config.py`: load YAML into a dataclass with the required fields (task, train, val and test paths, text and label columns, model size) and documented defaults (epochs 5, lr 5e-5, weight_decay 0.1, batch_size 8, seed 123, balance false, trainable last_blocks 1, max_length auto, label_names none). Verify with tests that a minimal config loads with defaults, and that a missing field or missing file raises an error naming it
- [x] 3.2 Implement the label mapping in `textclf/data.py` (D4): sorted string labels, `label_names`, and errors for an unknown evaluation label or fewer than 2 classes. Verify with tests for 3 string classes, integer labels with names, an unknown label, and a single class
- [x] 3.3 Implement `ClassificationDataset` (tokenize with tiktoken gpt2, truncate to max_length, skip and count empty texts, yield `(ids, label_id, length)`) and a collate function with per-batch right padding (50256). Verify with tests covering truncation, the skipped-row count, and that padded batches have the expected shapes and lengths
- [x] 3.4 Implement optional balancing (downsample to the smallest class, seeded, train split only). Verify with a test that 500/200/80 rows become 80/80/80 and that the result is identical across two calls with the same seed

## 4. Model

- [x] 4.1 Implement `textclf/model.py` `build_classifier(cfg, num_classes)`: GPTModel with BASE_CONFIG for the size, `download_and_load_gpt2` plus `load_weights_into_gpt`, an N-class `out_head`, and the freeze policy `head` / `last_blocks: K` / `all` (D3). Verify with a test (using a tiny model and mocked weights) that the `requires_grad` parameters match each policy
- [x] 4.2 Implement a pooled `predict_logits(model, ids, lengths)` helper shared by training eval and inference. Verify the padding-invariance spec scenario: the same text padded to 120 and to 1024 gives probabilities equal within 1e-5

## 5. Metrics and bundle

- [x] 5.1 Implement `textclf/metrics.py` (confusion matrix, per-class precision, recall, F1 and support, accuracy) with numpy. Verify against hand-computed values for a 3-class example, including a class that is never predicted (F1 = 0, no division error)
- [x] 5.2 Implement `textclf/bundle.py` save and load (D5): `model.pth`, `bundle.json`, `config.yaml` and `metrics.json`, written to a temporary directory and then `os.replace`d. Verify with tests that a round trip restores identical weights and labels, and that loading an incomplete bundle names the missing file

## 6. Training CLI

- [x] 6.1 Implement `python -m textclf.train <config.yaml> [--device] [--artifacts]`: validate the config, build the datasets and loaders (seeded generator), build the model, train with `train_classifier_simple`, evaluate the full splits, and save the bundle. Verify with an end-to-end test on a tiny 3-class CSV and a tiny model config that produces a complete bundle
- [x] 6.2 Write `tasks/spam.yaml` for the existing CSVs (`Text`/`Label`, `label_names: {0: ham, 1: spam}`, 124M, notebook hyperparameters). Verify that `python -m textclf.train tasks/spam.yaml` completes and that `artifacts/spam/metrics.json` reports test accuracy >= 0.94 (spam parity spec). If it falls short, tune only `tasks/spam.yaml` and record the change
- [x] 6.3 Verify reproducibility: train a tiny task twice on CPU with the same seed and assert identical test accuracy

## 7. Prediction

- [x] 7.1 Implement `textclf/predict.py` `Classifier.load(path, device)` and `.predict(texts)` returning a label and per-class probabilities, with no TensorFlow import. Verify with tests that probabilities sum to 1, output order matches input order, a batch result equals the single result within 1e-5, and that `import textclf.predict` does not import tensorflow
- [x] 7.2 Add the CLI `python -m textclf.predict --bundle artifacts/spam "text"` and verify it prints the label and probabilities with exit code 0
- [x] 7.3 Replace the body of `pre_trained_classifier_gpt_weights_model.py` with a thin call to `Classifier.load("artifacts/spam")` and verify it prints a label for its example prompt

## 8. FastAPI service

- [x] 8.1 Implement `textclf/server.py` `create_app(artifacts_dir, device)`: load bundles at startup, skip and log invalid ones, and expose `GET /health` and `GET /tasks` (D7). Verify with TestClient tests that `/tasks` lists the loaded tasks with labels and that a corrupt bundle is skipped without failing startup
- [x] 8.2 Implement `POST /classify/{task}` with request validation (exactly one of `text`/`texts`, non-empty strings, at most 64 texts). Verify with TestClient tests for a single text (200), multiple texts in order (200), an unknown task (404 listing available tasks) and each invalid body (422)
- [x] 8.3 Add the `python -m textclf.server` entry point defaulting to `127.0.0.1:8000`. Verify by starting it with `artifacts/spam` and running `curl -X POST localhost:8000/classify/spam -H 'content-type: application/json' -d '{"text":"Win a free prize now"}'`, which returns a label and probabilities

## 9. Wrap-up

- [x] 9.1 Prove generality with a second, non-spam dataset of 3 or more classes: add a small example task config, train it, and serve it alongside spam. Verify that `GET /tasks` lists both and `/classify/<task>` returns that task's labels
- [x] 9.2 Add a README section covering how to add a task (CSV format, config fields), train, predict, and run the server. Verify by following it from a clean shell
- [x] 9.3 Run the full `pytest` suite and confirm it passes
