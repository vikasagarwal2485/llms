"""GPT-2 classifier: the book's GPTModel with an N-class output head."""

import torch

from common import GPTModel
from classifier_model.common_classifier import pool_logits
from classifier_model.config import parse_trainable

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True,        # Query-key-value bias
}

MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def model_config(model_size):
    return {**BASE_CONFIG, **MODEL_CONFIGS[model_size]}


def create_classifier(model_cfg, num_classes):
    """GPTModel whose vocabulary head is replaced by Linear(emb_dim -> num_classes)."""
    model = GPTModel(model_cfg)
    model.out_head = torch.nn.Linear(in_features=model_cfg["emb_dim"], out_features=num_classes)
    return model


def load_pretrained_gpt(model_size, models_dir):
    """GPTModel with OpenAI GPT-2 weights; downloads missing files. Imports TensorFlow."""
    from common import load_weights_into_gpt
    from gpt_download import download_and_load_gpt2

    _, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)
    model = GPTModel(model_config(model_size))
    load_weights_into_gpt(model, params)
    return model


def apply_trainable(model, trainable):
    """Freeze everything, then unfreeze the head plus what `trainable` selects."""
    mode, k = parse_trainable(trainable)
    for param in model.parameters():
        param.requires_grad = mode == "all"
    for param in model.out_head.parameters():
        param.requires_grad = True
    if mode == "last_blocks":
        for block in list(model.trf_blocks)[len(model.trf_blocks) - k:] if k else []:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    return model


def build_classifier(cfg, num_classes):
    """Pre-trained GPT-2 with a fresh N-class head and the task's freeze policy."""
    model = load_pretrained_gpt(cfg.model_size, cfg.models_dir)
    torch.manual_seed(cfg.seed)
    model.out_head = torch.nn.Linear(in_features=model.final_norm.scale.shape[0], out_features=num_classes)
    return apply_trainable(model, cfg.trainable)


def predict_logits(model, input_ids, lengths):
    """Class logits pooled at each sequence's last real token."""
    return pool_logits(model(input_ids), lengths)


def config_of(model):
    """The GPTModel config needed to rebuild `model` (stored in bundles)."""
    first_attention = model.trf_blocks[0].att
    return {
        "vocab_size": model.tok_emb.num_embeddings,
        "context_length": model.pos_emb.num_embeddings,
        "drop_rate": 0.0,
        "qkv_bias": first_attention.W_query.bias is not None,
        "emb_dim": model.tok_emb.embedding_dim,
        "n_layers": len(model.trf_blocks),
        "n_heads": first_attention.num_heads,
    }


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
