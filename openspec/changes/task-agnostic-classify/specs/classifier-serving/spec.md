# Spec Delta

## ADDED Requirements

### Requirement: Task-agnostic HTTP classification
The HTTP service SHALL provide `POST /classify` accepting a JSON body with either a `text` string or a `texts` list of strings, and an optional `min_confidence` number between 0 and 1. The caller SHALL NOT name a task. The service SHALL run every loaded task on each input. For each input it SHALL return a mapping from task name to that task's result: the predicted label name, a `confidence` equal to the highest class probability, and a probability for every class of that task. When the body has `text`, `results` SHALL be that mapping. When the body has `texts`, `results` SHALL be a list of mappings in input order. When `min_confidence` is given, task results with a confidence below it SHALL be left out of the mapping.

#### Scenario: Classify single text across all tasks
- **WHEN** tasks `spam` and `ag_news` are loaded and a client posts `{"text": "URGENT! You won a £2000 bonus, call now"}` to `/classify`
- **THEN** the response status is 200 and `results` contains an entry for `spam` and an entry for `ag_news`, each with a label from that task's label set, a confidence, and one probability per class of that task

#### Scenario: Classify multiple texts
- **WHEN** a client posts `{"texts": ["hello", "free prize"]}` to `/classify`
- **THEN** the response status is 200 and `results` is a list of two mappings in input order, each containing every loaded task

#### Scenario: Confidence equals top probability
- **WHEN** any task result is returned
- **THEN** its `confidence` equals the largest value in its `probabilities` and its `label` is the class with that probability

#### Scenario: Low-confidence results filtered
- **WHEN** a client posts `{"text": "...", "min_confidence": 0.9}` and the `spam` result has confidence 0.99 while the `ag_news` result has confidence 0.41
- **THEN** the mapping contains `spam` and omits `ag_news`

#### Scenario: All results filtered
- **WHEN** every task result for an input is below `min_confidence`
- **THEN** the response status is 200 and that input's mapping is empty

#### Scenario: No tasks loaded
- **WHEN** the service has no loaded tasks and a client posts to `/classify`
- **THEN** the response status is 503 with a message stating that no tasks are loaded

#### Scenario: Invalid body
- **WHEN** the body has neither `text` nor `texts`, has both, has an empty or whitespace-only string, has more than 64 texts, or has a `min_confidence` outside 0 to 1
- **THEN** the response status is 422 with a message describing the problem

### Requirement: Multi-task results match single-task predictions
For every task and input, the label and probabilities returned by `POST /classify` SHALL equal, within 1e-5, those produced by loading that task's bundle alone and classifying the same input. This SHALL hold whatever the other loaded tasks are, including tasks with different maximum token lengths and tasks trained with different fine-tuning scopes.

#### Scenario: Equivalence with per-bundle prediction
- **WHEN** `spam` and `ag_news` are served together and a text is classified via `POST /classify`
- **THEN** each task's probabilities equal those from classifying the text with that task's bundle alone, within 1e-5

#### Scenario: Different maximum lengths
- **WHEN** a text tokenizes longer than one task's maximum length but shorter than another's
- **THEN** each task's result equals its single-bundle result, which truncates to that task's own maximum length

#### Scenario: Task with no shareable layers
- **WHEN** one loaded task was trained with all parameters trainable alongside tasks trained with the default scope
- **THEN** every task, including that one, is returned and matches its single-bundle result within 1e-5

## MODIFIED Requirements

### Requirement: HTTP task discovery
The HTTP service SHALL load every valid bundle under a configured artifacts directory at startup and SHALL expose the list of available tasks with their label sets and descriptions. A task whose bundle has no description SHALL be listed with a null description.

#### Scenario: List tasks
- **WHEN** a client sends `GET /tasks` and bundles `spam` and `news-topic` exist
- **THEN** the response lists both task names, each with its label names and description

#### Scenario: Bundle without description
- **WHEN** a bundle was trained without a description
- **THEN** `GET /tasks` lists that task with `description` set to null

#### Scenario: Invalid bundle at startup
- **WHEN** one bundle directory is incomplete or corrupt
- **THEN** the service starts, serves the valid bundles, and logs an error naming the skipped bundle

## REMOVED Requirements

### Requirement: HTTP classification
**Reason**: Callers, typically agents, should not need to know or choose a task. Every loaded task now runs on each request through `POST /classify`.
**Migration**: Replace `POST /classify/{task}` with `POST /classify` using the same `text`/`texts` body, and read `results[<task>]` (or `results[i][<task>]` for `texts`) from the response. A request to `/classify/{task}` now returns 404.
