import subprocess
import sys

import pytest
import torch

from textclf.bundle import save_bundle
from textclf.model import create_classifier
from textclf.predict import Classifier

LABELS = [{"id": i, "value": v, "name": v} for i, v in enumerate(["billing", "sales", "technical"])]
TEXTS = ["refund please", "the app keeps crashing when I open settings", "quote", "hi", "x " * 40,
         "pricing", "invoice wrong", "cannot log in", "buy seats", "error 500"]


@pytest.fixture
def bundle(tmp_path, tiny_config):
    torch.manual_seed(0)
    model = create_classifier(tiny_config, 3)
    return save_bundle(tmp_path, "support", model, LABELS, tiny_config, 32, 50256, {"task": "support"}, {})


def test_probabilities_and_order(bundle):
    clf = Classifier.load(bundle, "cpu")
    results = clf.predict(TEXTS)
    assert len(results) == 10
    for r in results:
        assert r["label"] in {"billing", "sales", "technical"}
        assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    singles = [clf.predict([t])[0] for t in TEXTS]
    for single, batched in zip(singles, results):
        assert single["label"] == batched["label"]
        for name in single["probabilities"]:
            assert single["probabilities"][name] == pytest.approx(batched["probabilities"][name], abs=1e-5)


def test_predict_does_not_import_tensorflow():
    code = "import sys, textclf.predict, textclf.server; print('tensorflow' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_cli(bundle):
    out = subprocess.run(
        [sys.executable, "-m", "textclf.predict", "--bundle", str(bundle), "--device", "cpu", "refund please"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "billing=" in out.stdout and "refund please" in out.stdout
