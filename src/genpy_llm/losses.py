"""Loss functions for GPT next-token prediction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from genpy_llm.config import LossConfig
from genpy_llm.vocabulary import Vocabulary, VocabularyError


class LossError(ValueError):
    """Raised when a GPT loss cannot be computed safely."""


class GPTCrossEntropyLoss(nn.Module):
    """Cross-entropy over raw GPT logits for next-token prediction."""

    def __init__(
        self,
        padding_idx: int | None = None,
        ignore_padding: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if padding_idx is not None and (
            not isinstance(padding_idx, int) or isinstance(padding_idx, bool) or padding_idx < 0
        ):
            raise LossError("padding_idx must be a non-negative integer or None.")
        if not isinstance(ignore_padding, bool):
            raise LossError("ignore_padding must be true or false.")
        if (
            not isinstance(label_smoothing, (int, float))
            or isinstance(label_smoothing, bool)
            or not 0.0 <= label_smoothing < 1.0
        ):
            raise LossError("label_smoothing must be at least 0.0 and less than 1.0.")
        self.padding_idx = padding_idx
        self.ignore_padding = ignore_padding
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return one scalar cross-entropy loss from raw logits and target IDs."""

        flattened_logits, flattened_targets = self._flatten_and_validate(logits, targets)
        ignore_index = (
            self.padding_idx if self.ignore_padding and self.padding_idx is not None else -100
        )
        if (
            self.ignore_padding
            and self.padding_idx is not None
            and _should_validate_tensor_values(flattened_targets)
            and bool((flattened_targets != self.padding_idx).sum().item() == 0)
        ):
            raise LossError("All target tokens are padding; cross-entropy loss is undefined.")
        return F.cross_entropy(
            flattened_logits,
            flattened_targets,
            ignore_index=ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def _flatten_and_validate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(logits, torch.Tensor):
            raise LossError("logits must be a torch.Tensor.")
        if not isinstance(targets, torch.Tensor):
            raise LossError("targets must be a torch.Tensor.")
        if logits.ndim not in {2, 3}:
            raise LossError("logits must have rank 2 or 3.")
        if targets.ndim not in {1, 2}:
            raise LossError("targets must have rank 1 or 2.")
        if logits.shape[-1] <= 0:
            raise LossError("logits vocabulary dimension must be greater than zero.")
        if targets.dtype != torch.long:
            raise LossError("targets must use torch.long dtype.")
        if not logits.is_floating_point():
            raise LossError("logits must use a floating-point dtype.")
        if _should_validate_tensor_values(logits) and not bool(torch.isfinite(logits).all().item()):
            raise LossError("logits must not contain NaN or infinite values.")

        if logits.ndim == 3 and targets.ndim == 2:
            if logits.shape[:2] != targets.shape:
                raise LossError("logits batch/sequence dimensions must match targets.")
            flattened_logits = logits.reshape(-1, logits.shape[-1])
            flattened_targets = targets.reshape(-1)
        elif logits.ndim == 2 and targets.ndim == 1:
            if logits.shape[0] != targets.shape[0]:
                raise LossError("flattened logits and targets must have the same item count.")
            flattened_logits = logits
            flattened_targets = targets
        else:
            raise LossError("logits and targets must both be batched or both be flattened.")

        self._validate_targets(flattened_targets, flattened_logits.shape[-1])
        return flattened_logits, flattened_targets

    def _validate_targets(self, targets: torch.Tensor, vocab_size: int) -> None:
        checked_targets = targets
        if self.ignore_padding and self.padding_idx is not None:
            checked_targets = targets[targets != self.padding_idx]
        if checked_targets.numel() == 0:
            return
        if _should_validate_tensor_values(checked_targets) and bool(
            (checked_targets < 0).any().item()
        ):
            raise LossError("targets must not contain negative token IDs.")
        if _should_validate_tensor_values(checked_targets) and bool(
            (checked_targets >= vocab_size).any().item()
        ):
            raise LossError("targets must be below the logits vocabulary size.")


def _should_validate_tensor_values(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "mps"


def create_loss_function(
    vocabulary_path: Path,
    loss_config: LossConfig,
) -> GPTCrossEntropyLoss:
    """Create configured GPT cross-entropy loss from a saved vocabulary."""

    if loss_config.type != "cross_entropy":
        raise LossError("Only cross_entropy loss is supported.")
    try:
        vocabulary = Vocabulary.load(vocabulary_path)
        padding_idx = vocabulary.pad_id
    except VocabularyError as exc:
        if loss_config.ignore_padding:
            raise LossError("Cannot ignore padding without a valid padding token.") from exc
        padding_idx = None
    return GPTCrossEntropyLoss(
        padding_idx=padding_idx,
        ignore_padding=loss_config.ignore_padding,
        label_smoothing=loss_config.label_smoothing,
    )


__all__ = ["GPTCrossEntropyLoss", "LossError", "create_loss_function"]
