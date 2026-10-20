import json
import logging

import pytest
import torch
from fastapi.testclient import TestClient

from textclf.bundle import save_bundle
from textclf.model import create_classifier
from textclf.server import create_app


def _labels(names):
    return [{"id": i, "value": n, "name": n} for i, n in enumerate(names)]


@pytest.fixture
def artifacts(tmp_path, tiny_config):
    torch.manual_seed(0)
    model = create_classifier(tiny_config, 2)
    save_bundle(tmp_path, "spam", model, _labels(["ham", "spam"]), tiny_config, 32, 50256,
                {"trainable": {"last_blocks": 1}}, {}, description="Detects SMS spam")
    model.out_head = torch.nn.Linear(tiny_config["emb_dim"], 3)
    save_bundle(tmp_path, "topic", model, _labels(["a", "b", "c"]), tiny_config, 32, 50256,
                {"trainable": {"last_blocks": 1}}, {})
    return tmp_path


@pytest.fixture
def client(artifacts):
    return TestClient(create_app(artifacts, "cpu"))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "tasks_loaded": 2}


def test_tasks_with_descriptions(client):
    tasks = {t["task"]: t for t in client.get("/tasks").json()["tasks"]}
    assert tasks["spam"] == {"task": "spam", "description": "Detects SMS spam", "labels": ["ham", "spam"]}
    assert tasks["topic"]["description"] is None and tasks["topic"]["labels"] == ["a", "b", "c"]


def test_corrupt_bundle_is_skipped(artifacts, caplog):
    (artifacts / "topic" / "model.pth").write_bytes(b"not a model")
    with caplog.at_level(logging.ERROR, logger="textclf.multitask"):
        client = TestClient(create_app(artifacts, "cpu"))
    assert client.get("/health").json()["tasks_loaded"] == 1
    assert "topic" in caplog.text


def test_classify_single_runs_every_task(client):
    r = client.post("/classify", json={"text": "URGENT! You won a bonus, call now"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert set(results) == {"spam", "topic"}
    assert results["spam"]["label"] in {"ham", "spam"}
    assert set(results["topic"]["probabilities"]) == {"a", "b", "c"}
    for res in results.values():
        assert res["confidence"] == max(res["probabilities"].values())


def test_classify_multiple_in_order(client):
    texts = ["hello", "free prize", "meeting at noon"]
    r = client.post("/classify", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    for text, res in zip(texts, results):
        assert set(res) == {"spam", "topic"}
        alone = client.post("/classify", json={"text": text}).json()["results"]
        for task in res:
            assert res[task]["label"] == alone[task]["label"]
            assert res[task]["probabilities"] == pytest.approx(alone[task]["probabilities"], abs=1e-5)


def test_min_confidence_filters(client):
    full = client.post("/classify", json={"text": "hello"}).json()["results"]
    threshold = (full["spam"]["confidence"] + full["topic"]["confidence"]) / 2
    keep = max(full, key=lambda t: full[t]["confidence"])
    r = client.post("/classify", json={"text": "hello", "min_confidence": threshold})
    assert list(r.json()["results"]) == [keep]


def test_min_confidence_can_filter_everything(client):
    r = client.post("/classify", json={"texts": ["hello"], "min_confidence": 1.0})
    assert r.status_code == 200
    assert r.json()["results"] == [{}]


def test_no_tasks_loaded(tmp_path):
    client = TestClient(create_app(tmp_path / "empty", "cpu"))
    r = client.post("/classify", json={"text": "hi"})
    assert r.status_code == 503
    assert "No tasks" in r.json()["detail"]


def test_per_task_endpoint_removed(client):
    assert client.post("/classify/spam", json={"text": "hi"}).status_code == 404


@pytest.mark.parametrize("body", [
    {},
    {"text": "a", "texts": ["b"]},
    {"text": ""},
    {"text": "   "},
    {"texts": []},
    {"texts": ["ok", ""]},
    {"texts": ["x"] * 65},
    {"text": "hi", "min_confidence": -0.1},
    {"text": "hi", "min_confidence": 1.5},
])
def test_invalid_body(client, body):
    assert client.post("/classify", json=body).status_code == 422
