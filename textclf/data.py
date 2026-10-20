"""Labeled CSV datasets for N-class text classification."""

import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN_ID = 50256  # <|endoftext|>, as in the book's SpamDataset


class DataError(ValueError):
    pass


def read_split(path, text_column, label_column):
    """Read a CSV split as a DataFrame with string columns `text` and `label`.

    Rows with empty or missing text are skipped; returns (df, num_skipped).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in (text_column, label_column):
        if column not in df.columns:
            raise DataError(f"Column '{column}' not found in {path}; columns are {list(df.columns)}")
    df = pd.DataFrame({"text": df[text_column].str.strip(), "label": df[label_column].str.strip()})
    keep = df["text"] != ""
    num_skipped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    if (df["label"] == "").any():
        raise DataError(f"{path} has rows with a missing label")
    return df, num_skipped


def _sort_key(value):
    # Integer labels sort numerically ("2" before "10"); anything else sorts as text
    try:
        return (0, int(value), "")
    except ValueError:
        return (1, 0, value)


def build_labels(train_labels, label_names=None):
    """Map the sorted distinct training labels to ids 0..N-1.

    Returns a list of {"id", "value", "name"} dicts, indexed by id.
    """
    values = sorted(set(train_labels), key=_sort_key)
    if len(values) < 2:
        raise DataError(f"At least two classes are required; the training split has {values}")
    label_names = label_names or {}
    return [{"id": i, "value": v, "name": label_names.get(v, v)} for i, v in enumerate(values)]


def encode_labels(labels, label_map, split):
    value_to_id = {entry["value"]: entry["id"] for entry in label_map}
    unknown = sorted(set(labels) - set(value_to_id), key=_sort_key)
    if unknown:
        raise DataError(f"Label(s) {unknown} in the {split} split do not appear in the training split")
    return [value_to_id[label] for label in labels]


def balance_classes(df, seed):
    """Downsample every class to the size of the smallest class."""
    smallest = df["label"].value_counts().min()
    parts = [group.sample(smallest, random_state=seed) for _, group in df.groupby("label", sort=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def encode_text(text, tokenizer, max_length):
    return tokenizer.encode(text)[:max_length]


class ClassificationDataset(Dataset):
    """Yields (token_ids, label_id, length) with no padding; see `collate`."""

    def __init__(self, texts, label_ids, tokenizer, max_length):
        self.encoded_texts = [encode_text(text, tokenizer, max_length) for text in texts]
        self.label_ids = list(label_ids)

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.label_ids[index], dtype=torch.long),
            torch.tensor(len(encoded), dtype=torch.long),
        )

    def __len__(self):
        return len(self.encoded_texts)


def collate(batch, pad_token_id=PAD_TOKEN_ID):
    """Right-pad each batch to its longest sequence; returns (input_ids, labels, lengths)."""
    ids, labels, lengths = zip(*batch)
    padded = torch.full((len(ids), max(lengths).item()), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(ids):
        padded[row, : len(seq)] = seq
    return padded, torch.stack(labels), torch.stack(lengths)


def longest_encoded_length(texts, tokenizer):
    return max(len(tokenizer.encode(text)) for text in texts)
