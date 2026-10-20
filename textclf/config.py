"""Task configuration: a YAML file per classification task."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

MODEL_SIZES = ("124M", "355M", "774M", "1558M")
REQUIRED_FIELDS = ("task", "train", "val", "test", "text_column", "label_column", "model_size")


class ConfigError(ValueError):
    pass


@dataclass
class TaskConfig:
    task: str
    train: str
    val: str
    test: str
    text_column: str
    label_column: str
    model_size: str
    # Free text telling callers (e.g. agents) what this task classifies
    description: str | None = None
    models_dir: str = "gpt2"
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    batch_size: int = 8
    seed: int = 123
    balance: bool = False
    # "head", "all", or {"last_blocks": K}; the final norm is trainable with last_blocks
    trainable: object = field(default_factory=lambda: {"last_blocks": 1})
    # None means: longest tokenized training text, capped at the model's context length
    max_length: int | None = None
    # Optional display names keyed by label value, e.g. {0: ham, 1: spam}
    label_names: dict | None = None
    eval_freq: int = 50
    eval_iter: int = 5

    def to_dict(self):
        return asdict(self)


def parse_trainable(value):
    """Return ("head", 0), ("all", 0) or ("last_blocks", K)."""
    if value in ("head", "all"):
        return value, 0
    if isinstance(value, dict) and set(value) == {"last_blocks"}:
        k = value["last_blocks"]
        if isinstance(k, int) and not isinstance(k, bool) and k >= 0:
            return "last_blocks", k
    raise ConfigError(f"Invalid 'trainable': {value!r}; use 'head', 'all' or {{last_blocks: K}}")


def config_from_dict(data):
    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping")
    missing = [name for name in REQUIRED_FIELDS if data.get(name) in (None, "")]
    if missing:
        raise ConfigError(f"Missing required config field(s): {', '.join(missing)}")
    known = {f.name for f in fields(TaskConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown config field(s): {', '.join(unknown)}")

    cfg = TaskConfig(**data)
    if cfg.model_size not in MODEL_SIZES:
        raise ConfigError(f"Invalid 'model_size': {cfg.model_size!r}; use one of {MODEL_SIZES}")
    parse_trainable(cfg.trainable)
    if cfg.description is not None and not isinstance(cfg.description, str):
        raise ConfigError("'description' must be a string")
    if cfg.label_names is not None:
        if not isinstance(cfg.label_names, dict):
            raise ConfigError("'label_names' must be a mapping of label value to name")
        cfg.label_names = {str(k): str(v) for k, v in cfg.label_names.items()}
    if cfg.max_length is not None and cfg.max_length < 1:
        raise ConfigError("'max_length' must be a positive integer")
    for name in ("train", "val", "test"):
        path = getattr(cfg, name)
        if not Path(path).is_file():
            raise ConfigError(f"Dataset file for '{name}' does not exist: {path}")
    return cfg


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_from_dict(data)
