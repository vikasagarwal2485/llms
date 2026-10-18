from importlib.metadata import version
import tiktoken
import torch
import torch.nn as nn

tokenizer = tiktoken.get_encoding("gpt2")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

from common_classifier import classify_review
from common import GPTModel

# Load the model state dictionary and run in evaluation mode

model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("pre_trained_classifier_gpt_weights_model.pth", map_location=device, weights_only=True))
model.eval();
# Model Inference to generate text with topk

torch.manual_seed(123)
print(classify_review(
    INPUT_PROMPT, model, tokenizer, device, max_length=model.pos_emb.weight.shape[0]
))
