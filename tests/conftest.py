import pytest
import torch

from common import GPTModel

# A tiny GPT with the real GPT-2 vocabulary, so tiktoken token ids stay valid
TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 16,
    "n_layers": 2,
    "n_heads": 2,
}


@pytest.fixture
def tiny_config():
    return dict(TINY_CONFIG)


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    model = GPTModel(tiny_config)
    model.out_head = torch.nn.Linear(tiny_config["emb_dim"], 3)
    return model
