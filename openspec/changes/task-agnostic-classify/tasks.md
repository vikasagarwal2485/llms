# Tasks

## 1. Task description

- [x] 1.1 Add optional `description: str | None` to `TaskConfig` in `textclf/config.py`, raising a `ConfigError` naming `description` when it isn't a string. Verify with tests in `tests/test_config.py`: a minimal config has `description` None, a string is accepted, and `description: 5` raises
- [x] 1.2 Store `description` in `bundle.json` via `save_bundle`, and pass it through from `run_training` in `textclf/train.py`. Verify with a test that a trained tiny bundle's `bundle.json` contains the description, and that a bundle with no `description` key loads with `meta.get("description") is None`
- [x] 1.3 Add `description` to `tasks/spam.yaml` and `tasks/ag_news.yaml`. Verify that `load_config` accepts both files

## 2. Multi-task execution

- [x] 2.1 Create `textclf/multitask.py` with a frozen-depth helper (D2: `head` gives `n_layers`, `{last_blocks: K}` gives `n_layers - K`, `all` gives 0). Verify with unit tests for each policy
- [x] 2.2 Implement `MultiTaskClassifier.load(artifacts_dir, device)`: read every bundle's config first, group tasks with depth >= 1 by `model_cfg`, compute each group's `D = min(d)`, then load bundles one at a time, keeping one trunk (`tok_emb`, `pos_emb`, `trf_blocks[:D]`) per group and a branch (`trf_blocks[D:]`, `final_norm`, `out_head`) per task (D3, D4). Skip and log invalid bundles. Verify with tests that two default-policy tiny bundles built from the same base end up in one group with `D = n_layers - 1`, and that an invalid bundle is skipped with an error naming it
- [x] 2.3 Implement the trunk weight check (D3): a task whose embeddings or shared blocks differ from the group's trunk moves to `solo` with a logged warning. Verify with a test where one bundle's block 0 weights are perturbed and that task ends up solo
- [x] 2.4 Implement `predict(texts)` (D5): tokenize once at `L = max(max_length)`, run the trunk once per group, run each branch, pool at `min(length, task max_length) - 1`, and run solo tasks as full models on the same batch. Return `[{task: {label, confidence, probabilities}}]` in input order, with `confidence = max(probabilities)`. Verify with a test that output order matches input and that every task appears for every text

## 3. Equivalence with per-bundle prediction

- [x] 3.1 Add `tests/test_multitask.py`. For tiny bundles mixing a default-policy task, a `head`-only task, an `all` (solo) task, and tasks with different `max_length`, assert that every task's probabilities from `MultiTaskClassifier.predict` equal `Classifier.load(bundle).predict` within 1e-5 on CPU, including a text longer than the smaller `max_length`
- [x] 3.2 Run the equivalence check on the real bundles: load `artifacts/spam` and `artifacts/ag_news` together on CPU, classify 20 texts from `sms_spam_collection_test.csv` and `datasets/ag_news/test.csv`, and confirm every task's probabilities match its single-bundle `Classifier` within 1e-5. Also confirm that both tasks share one group (log output shows one group, no solo tasks)

## 4. HTTP API

- [x] 4.1 Replace the request model in `textclf/server.py`: exactly one of `text` or `texts` (1-64 non-empty strings) plus optional `min_confidence` in [0, 1]. Verify with parametrized 422 tests covering a missing input, both inputs, an empty string, a whitespace-only string, an empty list, 65 texts, and `min_confidence` values of -0.1 and 1.5
- [x] 4.2 Add `POST /classify` backed by `MultiTaskClassifier`, returning a `results` mapping for `text` and a list of mappings for `texts`, applying the `min_confidence` filter, and returning 503 when no tasks are loaded. Verify with TestClient tests: every task present for a single text, list order for `texts`, a task dropped below the threshold, an empty mapping when everything is filtered, and 503 with an empty artifacts directory
- [x] 4.3 Remove `POST /classify/{task}` and its tests. Verify with a test that `POST /classify/spam` now returns 404
- [x] 4.4 Add `description` to each entry of `GET /tasks`, with null for bundles that have none. Verify with a TestClient test for one bundle with a description and one without
- [x] 4.5 Verify with the live server: run `python -m textclf.server --port 8765` against `artifacts/` and `curl -X POST localhost:8765/classify -d '{"text": "URGENT! You won a £2000 bonus, call now"}'` (with a JSON content-type header). Confirm the response has `spam` and `ag_news` results and that `min_confidence: 0.9` drops the low-confidence one

## 5. Docs and wrap-up

- [x] 5.1 Update `README.md`: the serving section shows `POST /classify` (single, batch, `min_confidence`), `GET /tasks` with descriptions, a note that `/classify/{task}` was removed, and the `description` config field. Verify the documented curl commands against the live server
- [x] 5.2 Run the full `pytest` suite and confirm it passes
