# Spec Delta

## Purpose

Fine-tunes the pre-trained GPT-2 open weights into a single-label text classifier for any number of classes, from a labeled CSV dataset and a task configuration, producing a self-describing model bundle with evaluation metrics.

## ADDED Requirements

### Requirement: Base weights acquisition
The system SHALL obtain the OpenAI GPT-2 open weights for the configured model size (124M, 355M, 774M or 1558M) into a local models directory, downloading only files that are missing, and SHALL fall back to the backup mirror when the primary URL fails.

#### Scenario: Weights missing locally
- **WHEN** training starts and the configured model size's weight files are absent
- **THEN** the missing files are downloaded and the base weights are loaded into the model

#### Scenario: Weights already present
- **WHEN** all weight files for the configured size exist locally
- **THEN** no download is performed and the local files are used

#### Scenario: Primary source unavailable
- **WHEN** a file download from the primary URL fails
- **THEN** the system retries from the backup URL and fails with an error naming the file only if both sources fail

### Requirement: Task configuration
Each classification task SHALL be defined by a configuration file specifying at least the task name, the train, validation and test dataset paths, the text and label column names, and the base model size. Training hyperparameters (epochs, learning rate, weight decay, batch size, maximum token length, random seed, balancing, trainable layers) SHALL be optional with documented defaults.

#### Scenario: Minimal config
- **WHEN** a config gives only the task name, dataset paths, column names and model size
- **THEN** training runs using the default hyperparameters

#### Scenario: Invalid config
- **WHEN** a config is missing a required field or references a dataset file that does not exist
- **THEN** training stops before loading any weights, with an error naming the field or file

### Requirement: General labeled dataset
The system SHALL accept CSV datasets with one text column and one label column, where labels are strings or integers. The set of classes SHALL be derived from the training split, sorted, and mapped to contiguous ids `0..N-1`, with N >= 2. For integer labels, the task config MAY give display names per label value.

#### Scenario: String labels with three classes
- **WHEN** the training split has labels `billing`, `technical` and `sales`
- **THEN** the model is trained with 3 output classes and the label mapping records all three names

#### Scenario: Integer labels with display names
- **WHEN** the labels are `0` and `1` and the config maps them to `ham` and `spam`
- **THEN** predictions report `ham` and `spam` as label names

#### Scenario: Unknown label in evaluation split
- **WHEN** the validation or test split contains a label that does not appear in the training split
- **THEN** training stops with an error naming the label and the split

#### Scenario: Fewer than two classes
- **WHEN** the training split contains only one distinct label
- **THEN** training stops with an error stating that at least two classes are required

#### Scenario: Missing or empty text
- **WHEN** a row has an empty or missing text value
- **THEN** that row is skipped and the number of skipped rows is reported

### Requirement: Optional class balancing
When balancing is enabled for a task, the system SHALL downsample every class in the training split to the size of the smallest class, reproducibly for a given seed. Validation and test splits SHALL NOT be balanced.

#### Scenario: Imbalanced three-class training set
- **WHEN** balancing is enabled and the classes have 500, 200 and 80 training rows
- **THEN** training uses 80 rows of each class

#### Scenario: Balancing disabled
- **WHEN** balancing is disabled
- **THEN** all training rows are used

### Requirement: Classification from the last real token
The classifier SHALL compute its class scores from the model output at the last non-padding token of each input, so the prediction for a text is independent of how much padding follows it. Inputs longer than the maximum token length SHALL be truncated to that length.

#### Scenario: Padding-invariant prediction
- **WHEN** the same text is classified once padded to 120 tokens and once padded to 1024 tokens
- **THEN** both produce the same label and probabilities equal within 1e-5

#### Scenario: Over-long input
- **WHEN** a text tokenizes to more tokens than the configured maximum length
- **THEN** only the first maximum-length tokens are used and classification succeeds

### Requirement: Configurable fine-tuning scope
The system SHALL replace the language-model output layer with an N-class classification layer and SHALL let the task config choose which parts are trained: only the classification layer, the classification layer plus the final normalization plus the last K transformer blocks, or all parameters. The default SHALL be the classification layer, the final normalization and the last 1 transformer block.

#### Scenario: Default scope
- **WHEN** a config does not specify the trainable layers
- **THEN** only the last transformer block, the final normalization and the classification layer receive gradient updates

#### Scenario: Head-only scope
- **WHEN** the config selects head-only training
- **THEN** all transformer and embedding weights equal the pre-trained values after training

### Requirement: Training run and model bundle
Training a task SHALL produce a model bundle directory containing the trained weights, the label mapping, the model and tokenizer settings needed for inference (model size, maximum token length, padding token), the resolved task config, and metrics. Bundle contents SHALL be sufficient to classify text without the original config or datasets.

#### Scenario: Successful training
- **WHEN** training of a task completes
- **THEN** the bundle directory for that task contains weights, label mapping, inference settings, the resolved config, and metrics

#### Scenario: Re-training an existing task
- **WHEN** a bundle already exists for the task and training completes again
- **THEN** the new bundle replaces the old one only after the new one is fully written

### Requirement: Evaluation metrics
After training, the system SHALL evaluate the model on the full train, validation and test splits and record per split the overall accuracy and, per class, precision, recall, F1 and support, plus a confusion matrix, along with training loss history and wall-clock training time.

#### Scenario: Metrics recorded
- **WHEN** training completes
- **THEN** the metrics include accuracy for all three splits, per-class precision, recall, F1 and support, and a confusion matrix labeled with class names

### Requirement: Reproducible training
Given the same config, data and seed on the same device type, training SHALL produce the same metrics.

#### Scenario: Repeat run
- **WHEN** a task is trained twice with identical config and seed on CPU
- **THEN** both runs report identical test accuracy

### Requirement: Spam task parity
The SMS spam dataset already in the repo SHALL be expressible as a task config, and with default settings SHALL reach at least 94% test accuracy, which preserves the existing 94.67% baseline within normal variance.

#### Scenario: Spam regression check
- **WHEN** the spam task is trained with its provided config
- **THEN** the reported test accuracy is at least 0.94
