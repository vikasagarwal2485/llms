import pytest

from textclf.config import ConfigError, load_config


def _write(tmp_path, body):
    for name in ("train.csv", "val.csv", "test.csv"):
        (tmp_path / name).write_text("text,label\nhello,a\n")
    path = tmp_path / "task.yaml"
    path.write_text(body)
    return path


MINIMAL = """\
task: demo
train: {d}/train.csv
val: {d}/val.csv
test: {d}/test.csv
text_column: text
label_column: label
model_size: 124M
"""


def test_minimal_config_uses_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL.format(d=tmp_path)))
    assert cfg.epochs == 5
    assert cfg.learning_rate == 5e-5
    assert cfg.weight_decay == 0.1
    assert cfg.batch_size == 8
    assert cfg.seed == 123
    assert cfg.balance is False
    assert cfg.trainable == {"last_blocks": 1}
    assert cfg.max_length is None


def test_missing_field_is_named(tmp_path):
    body = MINIMAL.format(d=tmp_path).replace("label_column: label\n", "")
    with pytest.raises(ConfigError, match="label_column"):
        load_config(_write(tmp_path, body))


def test_missing_dataset_file_is_named(tmp_path):
    body = MINIMAL.format(d=tmp_path).replace("test.csv", "nope.csv")
    with pytest.raises(ConfigError, match="nope.csv"):
        load_config(_write(tmp_path, body))


def test_label_names_keys_become_strings(tmp_path):
    body = MINIMAL.format(d=tmp_path) + "label_names: {0: ham, 1: spam}\n"
    cfg = load_config(_write(tmp_path, body))
    assert cfg.label_names == {"0": "ham", "1": "spam"}


@pytest.mark.parametrize("value", ["'everything'", "{last_blocks: -1}", "{blocks: 2}"])
def test_invalid_trainable(tmp_path, value):
    body = MINIMAL.format(d=tmp_path) + f"trainable: {value}\n"
    with pytest.raises(ConfigError, match="trainable"):
        load_config(_write(tmp_path, body))


def test_description_defaults_to_none(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL.format(d=tmp_path)))
    assert cfg.description is None


def test_description_string_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL.format(d=tmp_path) + "description: Detects SMS spam\n"))
    assert cfg.description == "Detects SMS spam"


def test_description_must_be_string(tmp_path):
    with pytest.raises(ConfigError, match="description"):
        load_config(_write(tmp_path, MINIMAL.format(d=tmp_path) + "description: 5\n"))
