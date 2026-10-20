# Spec Delta

## MODIFIED Requirements

### Requirement: Task configuration
Each classification task SHALL be defined by a configuration file specifying at least the task name, the train, validation and test dataset paths, the text and label column names, and the base model size. Training hyperparameters (epochs, learning rate, weight decay, batch size, maximum token length, random seed, balancing, trainable layers) SHALL be optional with documented defaults. An optional free-text `description` SHALL describe what the task classifies, for callers that discover tasks.

#### Scenario: Minimal config
- **WHEN** a config gives only the task name, dataset paths, column names and model size
- **THEN** training runs using the default hyperparameters and no description

#### Scenario: Config with description
- **WHEN** a config includes `description: "Detects SMS spam"`
- **THEN** training accepts it and the resulting bundle carries that description

#### Scenario: Invalid config
- **WHEN** a config is missing a required field, references a dataset file that does not exist, or has a `description` that is not a string
- **THEN** training stops before loading any weights, with an error naming the field or file

### Requirement: Training run and model bundle
Training a task SHALL produce a model bundle directory containing the trained weights, the label mapping, the model and tokenizer settings needed for inference (model size, maximum token length, padding token), the task description (or null when none was configured), the resolved task config, and metrics. Bundle contents SHALL be sufficient to classify text without the original config or datasets. Bundles produced before the description existed SHALL remain loadable and SHALL be treated as having no description.

#### Scenario: Successful training
- **WHEN** training of a task completes
- **THEN** the bundle directory for that task contains weights, label mapping, inference settings, the description, the resolved config, and metrics

#### Scenario: Re-training an existing task
- **WHEN** a bundle already exists for the task and training completes again
- **THEN** the new bundle replaces the old one only after the new one is fully written

#### Scenario: Older bundle without description
- **WHEN** a bundle written before this change is loaded
- **THEN** it loads successfully and reports no description
