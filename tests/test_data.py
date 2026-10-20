import pandas as pd
import pytest
import tiktoken
import torch

from classifier_model.data import (
    ClassificationDataset,
    DataError,
    balance_classes,
    build_labels,
    collate,
    encode_labels,
    read_split,
)

tokenizer = tiktoken.get_encoding("gpt2")


def test_three_string_classes():
    labels = build_labels(["sales", "billing", "technical", "billing"])
    assert [l["name"] for l in labels] == ["billing", "sales", "technical"]
    assert [l["id"] for l in labels] == [0, 1, 2]


def test_integer_labels_with_names_sort_numerically():
    labels = build_labels(["10", "2", "1"], {"1": "one", "2": "two"})
    assert [l["value"] for l in labels] == ["1", "2", "10"]
    assert [l["name"] for l in labels] == ["one", "two", "10"]


def test_unknown_label_in_eval_split():
    labels = build_labels(["a", "b"])
    with pytest.raises(DataError, match=r"\['c'\].*validation"):
        encode_labels(["a", "c"], labels, "validation")


def test_single_class_rejected():
    with pytest.raises(DataError, match="two classes"):
        build_labels(["a", "a"])


def test_empty_text_rows_skipped(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text('Text,Label\nhello,0\n"",1\n   ,1\nworld,1\n')
    df, skipped = read_split(path, "Text", "Label")
    assert skipped == 2
    assert df["text"].tolist() == ["hello", "world"]
    assert df["label"].tolist() == ["0", "1"]


def test_missing_column_is_named(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("Text,Label\nhello,0\n")
    with pytest.raises(DataError, match="'Category'"):
        read_split(path, "Text", "Category")


def test_truncation_and_collate_shapes():
    ds = ClassificationDataset(["hello world " * 50, "hi"], [0, 1], tokenizer, max_length=12)
    ids0, label0, len0 = ds[0]
    assert len0.item() == 12 and ids0.shape == (12,)
    padded, labels, lengths = collate([ds[0], ds[1]])
    assert padded.shape == (2, 12)
    assert lengths.tolist() == [12, 1]
    assert labels.tolist() == [0, 1]
    assert (padded[1, 1:] == 50256).all()


def test_balancing_is_seeded_and_downsamples():
    df = pd.DataFrame({"text": ["t"] * 780, "label": ["a"] * 500 + ["b"] * 200 + ["c"] * 80})
    df["text"] = [f"t{i}" for i in range(780)]
    first = balance_classes(df, seed=1)
    second = balance_classes(df, seed=1)
    assert first["label"].value_counts().to_dict() == {"a": 80, "b": 80, "c": 80}
    assert first.equals(second)
