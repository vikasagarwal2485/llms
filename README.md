# llms

Language models built on GPT-2, using the from-scratch `GPTModel` from *Build a Large Language Model From Scratch* and OpenAI's GPT-2 open weights.

The main component is **`classifier_model`**, which fine-tunes GPT-2 into a text classifier for any labeled dataset and serves every trained classifier through one local HTTP API. The GPT core it builds on (`common.py`, `gpt_download.py`) is shared, so the repo can grow a GPT text-generation model next to it (see [Extending](#extending-text-generation)).

## Repository layout

```
common.py              GPT core: GPTModel, generate_text_simple, load_weights_into_gpt  (shared)
gpt_download.py        downloads and loads OpenAI GPT-2 weights into gpt2/<size>/        (shared)
classifier_model/      text classification package
  config.py data.py model.py train.py metrics.py bundle.py predict.py multitask.py server.py
  common_classifier.py               training loop and last-token pooling helpers
  pre_trained_classifier_gpt_weights_model.py   minimal example: classify with the spam model
tasks/                 one YAML config per classification task (spam.yaml, ag_news.yaml)
data/                  datasets kept in git: SMS spam splits and raw TSV, the-verdict.txt (sample text for generation)
examples/              dataset preparation scripts (prepare_ag_news.py)
tests/                 pytest suite
openspec/              specs and change history (OpenSpec)

Generated and git-ignored: gpt2/ (base weights), artifacts/ (trained models), datasets/ (prepared example data)
```

## Setup

```bash
python3.11 -m venv .myenv
.myenv/bin/pip install -r requirements.txt
```

Commands below run from the repo root with `.myenv/bin/python`. The GPT-2 124M weights (about 500 MB) download into `gpt2/124M/` the first time a task trains.

## Text classification (`classifier_model`)

### 1. Prepare a dataset

You need three CSV files (train, validation, test), each with a text column and a label column. Labels can be strings (`billing`, `Sci/Tech`) or integers (`0`, `1`). The classes are the distinct labels in the training file, and there must be at least two. Rows with empty text are skipped. Any label in the validation or test file must also appear in the training file.

### 2. Write a task config

Create `tasks/<task>.yaml`. Paths are relative to the directory you run commands from.

```yaml
task: support            # bundle name; the key for this task in /classify results
description: Routes customer support tickets to billing, technical or sales.  # optional; shown in GET /tasks
train: data/support/train.csv
val: data/support/val.csv
test: data/support/test.csv
text_column: text
label_column: label
model_size: 124M         # 124M, 355M, 774M or 1558M
# Optional (defaults shown):
# epochs: 5
# learning_rate: 5.0e-5
# weight_decay: 0.1
# batch_size: 8
# seed: 123
# balance: false         # downsample each training class to the smallest one
# trainable: {last_blocks: 1}   # or: head, all
# max_length: null       # null = longest training text, capped at 1024 tokens
# label_names: {0: ham, 1: spam} # display names for label values
# eval_freq: 50          # steps between loss reports
# eval_iter: 5           # batches used per loss/accuracy report
```

Examples are in [`tasks/spam.yaml`](tasks/spam.yaml) and [`tasks/ag_news.yaml`](tasks/ag_news.yaml). Run `python examples/prepare_ag_news.py` before training the AG News task.

### 3. Train

```bash
.myenv/bin/python -m classifier_model.train tasks/spam.yaml        # --device cpu|mps|cuda, --artifacts DIR
```

This writes a self-contained bundle to `artifacts/<task>/`:

| File | Contents |
|---|---|
| `model.pth` | the fine-tuned weights |
| `bundle.json` | labels, description, model settings, max length |
| `config.yaml` | the resolved config |
| `metrics.json` | accuracy, per-class precision/recall/F1 and a confusion matrix for each split, plus loss history |

Retraining replaces the bundle only once the new one is fully written.

| Task | Classes | Test accuracy |
|---|---|---|
| spam | ham, spam | 97.0% |
| ag_news (4,000 training rows) | Business, Sci/Tech, Sports, World | 89.3% |

### 4. Predict

```bash
.myenv/bin/python -m classifier_model.predict --bundle artifacts/spam "Win a free prize now" "See you at lunch"
```

```python
from classifier_model.predict import Classifier
clf = Classifier.load("artifacts/spam")
clf.predict(["Win a free prize now"])  # [{"label": "spam", "probabilities": {"ham": ..., "spam": ...}}]
```

### 5. Serve

```bash
.myenv/bin/python -m classifier_model.server            # --artifacts artifacts --host 127.0.0.1 --port 8000
```

The server loads every bundle under `artifacts/` and listens on localhost only by default. Interactive docs are at `http://127.0.0.1:8000/docs`.

Callers don't pick a task. `POST /classify` runs **every** loaded task on the text and returns each task's answer, keyed by task name. An agent can call `GET /tasks` to learn what each task means.

```bash
curl localhost:8000/health
curl localhost:8000/tasks      # [{"task", "description", "labels"}, ...]

curl -X POST localhost:8000/classify -H 'content-type: application/json' \
  -d '{"text": "URGENT! You won a £2000 bonus, call now"}'
# {"results": {"spam":    {"label": "spam",     "confidence": 0.999, "probabilities": {"ham": 0.001, "spam": 0.999}},
#              "ag_news": {"label": "Business", "confidence": 0.31,  "probabilities": {"Business": 0.31, ...}}}}

curl -X POST localhost:8000/classify -H 'content-type: application/json' \
  -d '{"texts": ["Lakers win in overtime", "Are we still on for lunch?"], "min_confidence": 0.9}'
# {"results": [{"ag_news": {"label": "Sports", ...}}, {"spam": {"label": "ham", ...}}]}
```

- **Body:** exactly one of `text` or `texts` (at most 64). With `texts`, `results` is a list in input order.
- **`confidence`:** the top class probability.
- **`min_confidence`** (optional, 0 to 1): leaves out task results below the threshold. This is useful because every task answers every text, even off-topic ones.
- **Errors:** an invalid body returns 422, and a server with no trained bundles returns 503.
- **Removed endpoint:** `POST /classify/{task}` no longer exists (it returns 404). Use `POST /classify` and read `results[<task>]`.

Tasks whose frozen layers are identical (the default `trainable: {last_blocks: 1}`) share one pass through those layers, and each task then runs only its own last block and head. Results are identical to running each bundle on its own. Bundles that can't share run as full models.

### Tests

```bash
.myenv/bin/python -m pytest
```

## How it works

The classifier is the book's `GPTModel` loaded with OpenAI GPT-2 weights. Its 50,257-way vocabulary head is replaced by a `Linear(emb_dim -> N)` layer. By default only the last transformer block, the final layer norm and the new head are trained. Class scores come from the last *real* token of each text. Because GPT-2 only looks backward, the padding after that token has no effect, so predictions don't depend on padding length.

## Extending: text generation

The classifier swaps GPT-2's 50,257-way vocabulary head for an N-class head. A text-generation model keeps that head and reuses the same shared core:

- `common.GPTModel`, `load_weights_into_gpt` and `gpt_download.download_and_load_gpt2` give a pre-trained GPT-2 ready to generate.
- `common.generate_text_simple`, `text_to_token_ids` and `token_ids_to_text` cover greedy decoding.
- `common.GPTDatasetV1` / `create_dataloader_v1` build next-token training data, e.g. from `data/the-verdict.txt`.

To add one, create a sibling package (for example `generation_model/`) with its own train, generate and serve modules, following the `classifier_model` layout. Keep model-agnostic code in `common.py`.
