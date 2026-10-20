import torch
from torch.utils.data import DataLoader, TensorDataset

from common_classifier import (
    calc_accuracy_loader,
    calc_loss_batch,
    evaluate_model,
    train_classifier_simple,
)


def test_two_tuple_batch_pools_last_position(tiny_model):
    x = torch.randint(0, 50257, (4, 10))
    y = torch.tensor([0, 1, 2, 1])
    expected = torch.nn.functional.cross_entropy(tiny_model(x)[:, -1, :], y)
    loss = calc_loss_batch(x, y, tiny_model, "cpu")
    assert torch.allclose(loss, expected)


def test_three_tuple_batch_pools_at_last_real_token(tiny_model):
    x = torch.randint(0, 50257, (3, 10))
    y = torch.tensor([0, 1, 2])
    lengths = torch.tensor([4, 10, 1])
    logits = tiny_model(x)
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 3], logits[1, 9], logits[2, 0]]), y
    )
    loss = calc_loss_batch(x, y, tiny_model, "cpu", lengths)
    assert torch.allclose(loss, expected)


def test_accuracy_loader_accepts_both_batch_shapes(tiny_model):
    x = torch.randint(0, 50257, (6, 8))
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    lengths = torch.full((6,), 8)
    acc2 = calc_accuracy_loader(DataLoader(TensorDataset(x, y), batch_size=3), tiny_model, "cpu")
    acc3 = calc_accuracy_loader(DataLoader(TensorDataset(x, y, lengths), batch_size=3), tiny_model, "cpu")
    # With full-length sequences, last real token == last position
    assert acc2 == acc3


def test_training_loop_runs_with_three_tuple_batches(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 50257, (4, 8))
    y = torch.tensor([0, 1, 2, 0])
    lengths = torch.tensor([3, 8, 5, 2])
    loader = DataLoader(TensorDataset(x, y, lengths), batch_size=2)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    train_losses, val_losses, _, _, examples_seen = train_classifier_simple(
        tiny_model, loader, loader, optimizer, "cpu", num_epochs=1, eval_freq=1, eval_iter=1
    )
    assert examples_seen == 4
    assert len(train_losses) == 2
    train_loss, val_loss = evaluate_model(tiny_model, loader, loader, "cpu", eval_iter=1)
    assert train_loss == train_loss  # not NaN
