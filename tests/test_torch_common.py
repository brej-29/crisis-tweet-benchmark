"""Tests for the shared torch early-stopping loop (dtc.models.torch_common),
including the patience=None / restore_best_weights=False combination that
reproduces Protocol A's "no early stopping, fixed N epochs" replication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dtc.models.torch_common import fit_with_early_stopping


class _SqueezedLinear(nn.Module):
    """nn.Linear(4, 1) followed by squeeze(-1), so its output shape (batch,)
    matches the (batch,) target shape BCEWithLogitsLoss expects here -- the
    real models all do this squeeze internally (see dtc.models.neural_sequence
    etc.); this is a minimal stand-in, not a new pattern.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _make_loaders():
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,)).float()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=5), DataLoader(ds, batch_size=5)


def test_default_patience_can_stop_before_max_epochs_or_reach_it():
    train_loader, val_loader = _make_loaders()
    module = _SqueezedLinear()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    _, history = fit_with_early_stopping(
        module, train_loader, val_loader, optimizer, criterion, torch.device("cpu"), max_epochs=10, patience=3
    )
    assert 1 <= len(history.train_loss) <= 10
    assert len(history.train_loss) == len(history.val_loss)


def test_patience_none_always_runs_the_full_max_epochs():
    train_loader, val_loader = _make_loaders()
    module = _SqueezedLinear()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    _, history = fit_with_early_stopping(
        module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        torch.device("cpu"),
        max_epochs=5,
        patience=None,
    )
    assert len(history.train_loss) == 5
    assert history.stopped_epoch == 4


def test_restore_best_weights_false_skips_checkpoint_restoration():
    train_loader, val_loader = _make_loaders()
    module = _SqueezedLinear()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    fitted, history = fit_with_early_stopping(
        module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        torch.device("cpu"),
        max_epochs=3,
        patience=None,
        restore_best_weights=False,
    )
    assert fitted is module
    assert len(history.train_loss) == 3
