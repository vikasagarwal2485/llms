import copy
import logging

import pytest
import torch

from common import GPTModel
from textclf.bundle import save_bundle
from textclf.multitask import MultiTaskClassifier, frozen_depth
from textclf.predict import Classifier

TEXTS = [
    "refund please",
    "the app keeps crashing every time I open the settings page on my phone today",
    "hi",
    "Win a FREE prize now!!! Call 0800 123 456 to claim your reward before midnight",
    "quarterly earnings beat expectations",
]


@pytest.fixture
def base(tiny_config):
    torch.manual_seed(0)
    return GPTModel(tiny_config)


def make_bundle(artifacts, base, cfg, task, labels, trainable, max_length=32, seed=1, description=None):
    """A bundle as training would produce it: pre-trained base, new head, only `trainable` layers changed."""
    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], len(labels))
    if trainable == "all":
        changed = list(model.parameters())
    elif isinstance(trainable, dict):
        k = trainable["last_blocks"]
        changed = [p for b in list(model.trf_blocks)[len(model.trf_blocks) - k:] for p in b.parameters()]
        changed += list(model.final_norm.parameters())
    else:
        changed = []
    with torch.no_grad():
        for p in changed:
            p.add_(0.05 * torch.randn_like(p))
    label_map = [{"id": i, "value": n, "name": n} for i, n in enumerate(labels)]
    return save_bundle(artifacts, task, model, label_map, cfg, max_length, 50256,
                       {"task": task, "trainable": trainable}, {}, description=description)


def test_frozen_depth():
    assert frozen_depth("head", 12) == 12
    assert frozen_depth({"last_blocks": 1}, 12) == 11
    assert frozen_depth({"last_blocks": 3}, 12) == 9
    assert frozen_depth({"last_blocks": 0}, 12) == 12
    assert frozen_depth("all", 12) == 0


def test_default_policy_tasks_share_one_group(tmp_path, base, tiny_config):
    make_bundle(tmp_path, base, tiny_config, "spam", ["ham", "spam"], {"last_blocks": 1}, seed=1)
    make_bundle(tmp_path, base, tiny_config, "topic", ["a", "b", "c"], {"last_blocks": 1}, seed=2)
    clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert len(clf.groups) == 1 and not clf.solo
    assert clf.groups[0].depth == tiny_config["n_layers"] - 1
    assert sorted(clf.groups[0].branches) == ["spam", "topic"]


def test_all_policy_runs_solo(tmp_path, base, tiny_config):
    make_bundle(tmp_path, base, tiny_config, "spam", ["ham", "spam"], {"last_blocks": 1})
    make_bundle(tmp_path, base, tiny_config, "full", ["x", "y"], "all", seed=3)
    clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert list(clf.solo) == ["full"]
    assert sorted(clf.groups[0].branches) == ["spam"]


def test_perturbed_shared_layers_run_solo(tmp_path, base, tiny_config, caplog):
    make_bundle(tmp_path, base, tiny_config, "a_task", ["x", "y"], {"last_blocks": 1})
    other_base = copy.deepcopy(base)
    with torch.no_grad():
        other_base.trf_blocks[0].att.W_query.weight.add_(1e-3)
    make_bundle(tmp_path, other_base, tiny_config, "b_task", ["x", "y"], {"last_blocks": 1})
    with caplog.at_level(logging.WARNING, logger="textclf.multitask"):
        clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert list(clf.solo) == ["b_task"]
    assert "b_task" in caplog.text


def test_invalid_bundle_skipped(tmp_path, base, tiny_config, caplog):
    make_bundle(tmp_path, base, tiny_config, "good", ["x", "y"], {"last_blocks": 1})
    bad = make_bundle(tmp_path, base, tiny_config, "bad", ["x", "y"], {"last_blocks": 1})
    (bad / "model.pth").write_bytes(b"corrupt")
    with caplog.at_level(logging.ERROR, logger="textclf.multitask"):
        clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert list(clf.tasks) == ["good"]
    assert "bad" in caplog.text


def test_predict_order_and_every_task(tmp_path, base, tiny_config):
    make_bundle(tmp_path, base, tiny_config, "spam", ["ham", "spam"], {"last_blocks": 1}, description="d")
    make_bundle(tmp_path, base, tiny_config, "topic", ["a", "b", "c"], {"last_blocks": 1}, seed=2)
    clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert clf.tasks["spam"].description == "d" and clf.tasks["topic"].description is None
    results = clf.predict(TEXTS)
    assert len(results) == len(TEXTS)
    for res in results:
        assert set(res) == {"spam", "topic"}
        for r in res.values():
            assert r["confidence"] == max(r["probabilities"].values())
            assert r["probabilities"][r["label"]] == r["confidence"]
    # Order: each batched result equals classifying that text alone
    for text, res in zip(TEXTS, results):
        alone = clf.predict([text])[0]
        for name in res:
            assert res[name]["probabilities"] == pytest.approx(alone[name]["probabilities"], abs=1e-5)


def test_equivalence_with_single_bundles(tmp_path, base, tiny_config):
    bundles = {
        "default_short": make_bundle(tmp_path, base, tiny_config, "default_short", ["a", "b"],
                                     {"last_blocks": 1}, max_length=8, seed=1),
        "default_long": make_bundle(tmp_path, base, tiny_config, "default_long", ["a", "b", "c"],
                                    {"last_blocks": 1}, max_length=40, seed=2),
        "head_only": make_bundle(tmp_path, base, tiny_config, "head_only", ["p", "q"], "head", seed=3),
        "everything": make_bundle(tmp_path, base, tiny_config, "everything", ["u", "v", "w", "z"], "all", seed=4),
    }
    clf = MultiTaskClassifier.load(tmp_path, "cpu")
    assert sorted(clf.groups[0].branches) == ["default_long", "default_short", "head_only"]
    assert list(clf.solo) == ["everything"]
    assert len(clf.tokenizer.encode(TEXTS[1])) > 8  # longer than default_short's max_length

    multi = clf.predict(TEXTS)
    for name, path in bundles.items():
        single = Classifier.load(path, "cpu").predict(TEXTS)
        for m, s in zip(multi, single):
            assert m[name]["label"] == s["label"]
            for label, p in s["probabilities"].items():
                assert m[name]["probabilities"][label] == pytest.approx(p, abs=1e-5)
