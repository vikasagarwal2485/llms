import tiktoken
import torch

from classifier_model import model as model_module
from classifier_model.config import TaskConfig
from classifier_model.data import ClassificationDataset, PAD_TOKEN_ID
from classifier_model.model import apply_trainable, create_classifier, model_config, predict_logits


def _trainable_names(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_head_only(tiny_config):
    model = apply_trainable(create_classifier(tiny_config, 3), "head")
    assert _trainable_names(model) == {"out_head.weight", "out_head.bias"}


def test_default_last_block(tiny_config):
    model = apply_trainable(create_classifier(tiny_config, 3), {"last_blocks": 1})
    names = _trainable_names(model)
    assert {"out_head.weight", "out_head.bias", "final_norm.scale", "final_norm.shift"} <= names
    assert all(n.startswith(("out_head", "final_norm", "trf_blocks.1.")) for n in names)
    assert any(n.startswith("trf_blocks.1.") for n in names)


def test_all(tiny_config):
    model = apply_trainable(create_classifier(tiny_config, 3), "all")
    assert all(p.requires_grad for p in model.parameters())


def test_build_classifier_uses_pretrained_and_policy(tiny_config, tmp_path, monkeypatch):
    from common import GPTModel

    monkeypatch.setattr(model_module, "load_pretrained_gpt", lambda size, d: GPTModel(tiny_config))
    cfg = TaskConfig(task="t", train="a", val="b", test="c", text_column="x",
                     label_column="y", model_size="124M", trainable="head")
    model = model_module.build_classifier(cfg, num_classes=4)
    assert model.out_head.out_features == 4
    assert _trainable_names(model) == {"out_head.weight", "out_head.bias"}


def test_padding_invariance_120_vs_1024():
    torch.manual_seed(0)
    model = create_classifier(model_config("124M"), 2).eval()
    ids, _, length = ClassificationDataset(["Win a free prize now"], [0], tiktoken.get_encoding("gpt2"), 1024)[0]

    def probs(pad_to):
        padded = torch.full((1, pad_to), PAD_TOKEN_ID, dtype=torch.long)
        padded[0, : len(ids)] = ids
        with torch.inference_mode():
            return torch.softmax(predict_logits(model, padded, length.unsqueeze(0)), dim=-1)

    assert torch.allclose(probs(120), probs(1024), atol=1e-5)
