"""Shared torch training utilities: seeding, device selection, and a
generic early-stopping training loop reused by every torch-based model in
dtc.models.* (Hard Rule 4: device-agnostic; Hard Rule 5: determinism
discipline).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seeds python/numpy/torch for one run.

    `torch.use_deterministic_algorithms(True, warn_only=True)` is used
    rather than the strict (error-raising) form: some ops (mainly
    CUDA-specific ones) have no deterministic implementation, and per Hard
    Rule 5 the project documents that residual nondeterminism (see
    docs/DECISIONS.md) instead of claiming bit-exactness it can't back up on
    GPU runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EarlyStoppingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_epoch: int = -1


StepFn = Callable[[torch.nn.Module, tuple], torch.Tensor]


def fit_with_early_stopping_generic(
    module: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_step_fn: StepFn,
    val_step_fn: StepFn,
    *,
    max_epochs: int = 30,
    patience: int | None = 3,
    restore_best_weights: bool = True,
) -> tuple[torch.nn.Module, EarlyStoppingHistory]:
    """Training loop, generic over how a batch maps to a loss.

    `train_step_fn(module, batch) -> loss` and `val_step_fn(module, batch) ->
    loss` let callers with different batch shapes/forward signatures (plain
    (X, y) tuples vs. HF's (input_ids, attention_mask, labels)) share one
    implementation. `max_epochs` must be high enough that stopping actually
    binds when `patience` is set (Hard Rule/PLAN 1.2).

    `patience=None` disables early stopping entirely (always runs
    `max_epochs` epochs) and `restore_best_weights=False` keeps the final
    epoch's weights instead of the best-val-loss checkpoint -- together
    these reproduce Protocol A's "no early stopping, fixed N epochs"
    replication of the original (flawed) training procedure (docs/PLAN.md
    1.3), as opposed to Protocol B's default (patience=3,
    restore_best_weights=True).
    """
    history = EarlyStoppingHistory()
    best_val_loss = float("inf")
    best_state = copy.deepcopy(module.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        module.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = train_step_fn(module, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        module.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(val_step_fn(module, batch).item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        history.train_loss.append(mean_train_loss)
        history.val_loss.append(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(module.state_dict())
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                history.stopped_epoch = epoch
                break
    else:
        history.stopped_epoch = max_epochs - 1

    if restore_best_weights:
        module.load_state_dict(best_state)
    return module, history


def fit_with_early_stopping(
    module: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    max_epochs: int = 30,
    patience: int | None = 3,
    restore_best_weights: bool = True,
) -> tuple[torch.nn.Module, EarlyStoppingHistory]:
    """Convenience wrapper for the common case: batches are (X, y) tuples and
    `module(X)` returns a 1-D logit tensor matching `y`'s shape.
    """
    module.to(device)

    def step(m: torch.nn.Module, batch: tuple) -> torch.Tensor:
        X, y = batch
        X, y = X.to(device), y.to(device)
        return criterion(m(X), y)

    return fit_with_early_stopping_generic(
        module,
        train_loader,
        val_loader,
        optimizer,
        step,
        step,
        max_epochs=max_epochs,
        patience=patience,
        restore_best_weights=restore_best_weights,
    )
