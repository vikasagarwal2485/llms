"""Classify text with a trained model bundle (no TensorFlow needed).

Usage: python -m textclf.predict --bundle artifacts/<task> "some text" ["more text" ...]
"""

import argparse
import sys

import tiktoken
import torch

from textclf.bundle import BundleError, load_bundle
from textclf.data import collate, encode_text
from textclf.model import predict_logits, select_device


class Classifier:
    def __init__(self, meta, model, device):
        self.meta = meta
        self.model = model
        self.device = device
        self.task = meta["task"]
        self.description = meta.get("description")  # absent in bundles trained before descriptions
        self.label_names = [label["name"] for label in meta["labels"]]
        self.max_length = meta["max_length"]
        self.pad_token_id = meta["pad_token_id"]
        self.tokenizer = tiktoken.get_encoding(meta["tokenizer"])

    @classmethod
    def load(cls, path, device="auto"):
        device = select_device(device)
        meta, model = load_bundle(path, device)
        return cls(meta, model, device)

    def predict(self, texts, batch_size=32):
        """Return [{"label": name, "probabilities": {name: p, ...}}, ...] in input order."""
        results = []
        for start in range(0, len(texts), batch_size):
            batch = []
            for text in texts[start:start + batch_size]:
                ids = encode_text(text, self.tokenizer, self.max_length)
                batch.append((torch.tensor(ids), torch.tensor(0), torch.tensor(len(ids))))
            input_ids, _, lengths = collate(batch, self.pad_token_id)
            with torch.inference_mode():
                logits = predict_logits(self.model, input_ids.to(self.device), lengths.to(self.device))
                probs = torch.softmax(logits.float(), dim=-1).cpu()
            for row in probs:
                results.append({
                    "label": self.label_names[int(row.argmax())],
                    "probabilities": dict(zip(self.label_names, row.tolist())),
                })
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="Path to a model bundle, e.g. artifacts/spam")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("texts", nargs="+", help="Text(s) to classify")
    args = parser.parse_args(argv)
    try:
        classifier = Classifier.load(args.bundle, args.device)
    except BundleError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    for text, result in zip(args.texts, classifier.predict(args.texts)):
        probabilities = ", ".join(f"{name}={p:.4f}" for name, p in result["probabilities"].items())
        print(f"{result['label']}\t[{probabilities}]\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
