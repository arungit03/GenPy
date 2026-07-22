"""Phase 7 supervised fine-tuning evaluation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    """Validation metrics for supervised causal language modeling."""

    loss: float
    perplexity: float
    tokens: int
    batches: int


def evaluation_metrics(loss: float, tokens: int, batches: int) -> EvaluationMetrics:
    """Build stable validation metrics from average loss."""

    perplexity = math.exp(min(20.0, loss)) if loss > 0 else 0.0
    return EvaluationMetrics(
        loss=float(loss),
        perplexity=float(perplexity),
        tokens=int(tokens),
        batches=int(batches),
    )


__all__ = ["EvaluationMetrics", "evaluation_metrics"]
