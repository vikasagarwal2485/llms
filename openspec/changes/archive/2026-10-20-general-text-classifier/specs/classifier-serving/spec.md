# Spec Delta

## Purpose

Loads trained text classifier model bundles and classifies text with them, both in-process and through a local HTTP API that serves multiple tasks at once.

## ADDED Requirements

### Requirement: In-process prediction
The system SHALL load a model bundle by path and classify one text or a list of texts, returning for each text the predicted label name and a probability for every class, where the probabilities sum to 1 within 1e-5.

#### Scenario: Single text
- **WHEN** a loaded spam bundle classifies "You are a winner, claim your $1000 cash prize now"
- **THEN** the result contains a label from the bundle's label set and one probability per class

#### Scenario: Batch of texts
- **WHEN** a list of 10 texts is classified
- **THEN** 10 results are returned in the same order as the input

#### Scenario: Batch equals single
- **WHEN** a text is classified alone and as part of a batch
- **THEN** both results have the same label and probabilities equal within 1e-5

#### Scenario: Invalid bundle
- **WHEN** the bundle path does not exist or a required bundle file is missing
- **THEN** loading fails with an error naming the path or missing file

### Requirement: Command-line prediction
The system SHALL provide a command that classifies text given as an argument with a named bundle and prints the label and class probabilities.

#### Scenario: CLI classify
- **WHEN** the user runs the predict command with a bundle and a text
- **THEN** the label and the per-class probabilities are printed and the exit code is 0

### Requirement: HTTP task discovery
The HTTP service SHALL load every valid bundle under a configured artifacts directory at startup and SHALL expose the list of available tasks with their label sets.

#### Scenario: List tasks
- **WHEN** a client sends `GET /tasks` and bundles `spam` and `news-topic` exist
- **THEN** the response lists both task names, each with its label names

#### Scenario: Invalid bundle at startup
- **WHEN** one bundle directory is incomplete or corrupt
- **THEN** the service starts, serves the valid bundles, and logs an error naming the skipped bundle

### Requirement: HTTP classification
The HTTP service SHALL provide `POST /classify/{task}` accepting a JSON body with either a `text` string or a `texts` list of strings, and SHALL respond with the predicted label and per-class probabilities for each input, in input order.

#### Scenario: Classify single text
- **WHEN** a client posts `{"text": "Win a free iPhone now"}` to `/classify/spam`
- **THEN** the response status is 200 and the body contains the task name, the predicted label, and a probability for each class

#### Scenario: Classify multiple texts
- **WHEN** a client posts `{"texts": ["hello", "free prize"]}` to `/classify/spam`
- **THEN** the response status is 200 and contains two results in input order

#### Scenario: Unknown task
- **WHEN** a client posts to `/classify/unknown-task`
- **THEN** the response status is 404 and the body names the available tasks

#### Scenario: Invalid body
- **WHEN** the body has neither `text` nor `texts`, has both, has an empty string, or has more than the maximum batch size of texts
- **THEN** the response status is 422 with a message describing the problem

### Requirement: Health check
The HTTP service SHALL provide `GET /health`, returning status 200 with the number of loaded tasks.

#### Scenario: Service healthy
- **WHEN** a client sends `GET /health`
- **THEN** the response status is 200 and includes the count of loaded tasks

### Requirement: Local-only default binding
The HTTP service SHALL listen on `127.0.0.1` by default; the host and port SHALL be configurable.

#### Scenario: Default start
- **WHEN** the service is started without host options
- **THEN** it accepts connections on 127.0.0.1 only
