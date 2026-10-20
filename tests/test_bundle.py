import pytest
import torch

from classifier_model.bundle import BundleError, load_bundle, save_bundle
from classifier_model.model import create_classifier

LABELS = [{"id": 0, "value": "a", "name": "A"}, {"id": 1, "value": "b", "name": "B"}]


def _save(tmp_path, cfg, seed=0):
    torch.manual_seed(seed)
    model = create_classifier(cfg, 2)
    path = save_bundle(tmp_path, "demo", model, LABELS, cfg, 32, 50256, {"task": "demo"}, {"test": {}})
    return model, path


def test_round_trip(tmp_path, tiny_config):
    model, path = _save(tmp_path, tiny_config)
    meta, loaded = load_bundle(path)
    assert meta["labels"] == LABELS and meta["max_length"] == 32
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_retrain_replaces_bundle_without_leftovers(tmp_path, tiny_config):
    _save(tmp_path, tiny_config, seed=0)
    model2, path = _save(tmp_path, tiny_config, seed=1)
    _, loaded = load_bundle(path)
    assert torch.equal(loaded.out_head.weight, model2.out_head.weight)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["demo"]


def test_missing_file_is_named(tmp_path, tiny_config):
    _, path = _save(tmp_path, tiny_config)
    (path / "bundle.json").unlink()
    with pytest.raises(BundleError, match="bundle.json"):
        load_bundle(path)


def test_missing_dir_is_named(tmp_path):
    with pytest.raises(BundleError, match="nowhere"):
        load_bundle(tmp_path / "nowhere")


def test_bundle_without_description_key_loads(tmp_path, tiny_config):
    import json

    _, path = _save(tmp_path, tiny_config)
    meta_path = path / "bundle.json"
    meta = json.loads(meta_path.read_text())
    del meta["description"]
    meta_path.write_text(json.dumps(meta))
    loaded_meta, _ = load_bundle(path)
    assert loaded_meta.get("description") is None
