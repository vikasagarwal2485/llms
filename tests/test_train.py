import pytest

from common import GPTModel
from textclf import model as model_module
from textclf.bundle import REQUIRED_FILES, load_bundle
from textclf.config import TaskConfig
from textclf.train import run_training

ROWS = {
    "billing": ["my invoice is wrong", "refund my payment", "charged twice this month"],
    "technical": ["app crashes on start", "cannot log in", "error 500 on upload"],
    "sales": ["pricing for teams", "buy more seats", "enterprise plan quote"],
}


@pytest.fixture
def three_class_task(tmp_path, tiny_config, monkeypatch):
    monkeypatch.setattr(model_module, "load_pretrained_gpt", lambda size, d: GPTModel(tiny_config))
    for split in ("train", "val", "test"):
        lines = ["text,label"] + [f"{t},{label}" for label, texts in ROWS.items() for t in texts]
        (tmp_path / f"{split}.csv").write_text("\n".join(lines) + "\n")
    return TaskConfig(
        task="support", train=str(tmp_path / "train.csv"), val=str(tmp_path / "val.csv"),
        test=str(tmp_path / "test.csv"), text_column="text", label_column="label",
        model_size="124M", description="Routes support tickets", epochs=2, batch_size=4, eval_freq=1, eval_iter=1,
    )


def test_end_to_end_bundle(three_class_task, tmp_path):
    bundle_path, metrics = run_training(three_class_task, tmp_path / "artifacts", device="cpu", log=lambda *_: None)
    assert all((bundle_path / name).is_file() for name in REQUIRED_FILES)
    meta, model = load_bundle(bundle_path)
    assert [l["name"] for l in meta["labels"]] == ["billing", "sales", "technical"]
    assert model.out_head.out_features == 3
    assert meta["description"] == "Routes support tickets"
    test = metrics["splits"]["test"]
    assert set(test) == {"accuracy", "per_class", "confusion_matrix"}
    assert test["confusion_matrix"]["labels"] == ["billing", "sales", "technical"]
    assert set(metrics["splits"]) == {"train", "val", "test"}


def test_reproducible_on_cpu(three_class_task, tmp_path):
    quiet = lambda *_: None
    _, first = run_training(three_class_task, tmp_path / "a1", device="cpu", log=quiet)
    _, second = run_training(three_class_task, tmp_path / "a2", device="cpu", log=quiet)
    assert first["splits"]["test"] == second["splits"]["test"]
    assert first["history"]["train_losses"] == second["history"]["train_losses"]
