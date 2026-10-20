# Proposal

## Why

The HTTP service makes callers choose a model up front (`POST /classify/{task}`). An agent calling it usually doesn't know which classifiers exist or which one fits a text, and the trained tasks answer *different* questions about the same text: a message is both spam-or-ham *and* about some topic. The agent needs one endpoint that takes only text and returns every classifier's answer, so it can use the ones that matter.

This is cheap to provide. With the default fine-tuning policy, every task's bundle keeps the GPT-2 embeddings and the first 11 blocks frozen. Comparing the `spam` and `ag_news` bundles shows 129.9M of their parameters are identical and only 7.1M differ (the last block, the final norm and the head). Running all tasks can share one pass through those common layers.

## What Changes

- **BREAKING**: remove `POST /classify/{task}`. Callers move to `POST /classify` and read the task they want from the response.
- Add task-agnostic `POST /classify`. It accepts `text` or `texts` (at most 64), runs **every** loaded task, and returns results keyed by task name. Each result has a label, a confidence (the top-class probability) and per-class probabilities.
- Add an optional `min_confidence` request field that leaves out task results whose confidence is below the threshold.
- Return `503` from `POST /classify` when no tasks are loaded.
- Extend `GET /tasks` with each task's `description`, so an agent can discover what each classifier does without knowing task names in advance.
- Add an optional `description` field to the task config, stored in the model bundle. Bundles trained before this change still load, with no description.
- Shared-layer execution in the server. Bundles whose frozen layers are identical share one pass through those layers, and each task runs only its own trained layers. A bundle whose layers can't be shared still runs as a full model. Results are identical (within 1e-5) to running each bundle on its own.

Non-goals: picking a single "best" task automatically, merging label sets across tasks, per-request task filters, changes to the per-bundle prediction CLI, and changes to training behavior beyond the new `description` field.

## Capabilities

### New Capabilities
<!-- None -->

### Modified Capabilities
- `classifier-serving`: the per-task classify endpoint is replaced by a task-agnostic one that runs all tasks; task discovery adds descriptions; serving shares common layers across tasks.
- `classifier-training`: the task config accepts an optional description, and the model bundle stores it.

Sequencing: both capabilities are introduced by the `general-text-classifier` change, which is implemented but not yet archived. Archive that change (which creates `openspec/specs/classifier-serving` and `openspec/specs/classifier-training`) before archiving this one.

## Impact

- **Code**: `textclf/server.py` (new endpoint, removal of the old one, startup grouping); a new multi-task execution module in `textclf/`; `textclf/config.py` and `textclf/train.py` (description); `textclf/bundle.py` (store and read description); tests in `tests/test_server.py` plus new equivalence tests; `README.md` (API section).
- **API**: breaking for any client of `POST /classify/{task}`. Inside this repo, the only clients are the tests and the README.
- **Existing bundles**: `artifacts/spam` and `artifacts/ag_news` keep working without retraining. To add descriptions, put them in the task configs and retrain.
- **Resources**: the server keeps one copy of the shared layers plus about 28 MB per task, instead of about 500 MB per task, and each request does one shared pass plus one transformer block per task.
