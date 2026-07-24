"""Dependency-free shims for standard-library features from newer Python versions.

Kept separate from utils.py (which pulls in numpy/torch) so lightweight modules
such as the tokenizer and corpus pipelines stay import-light under Python 3.9.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def zip_strict(*iterables: Iterable[object]) -> Iterator[tuple]:
    """Python 3.9-compatible equivalent of ``zip(*iterables, strict=True)``."""

    sentinel = object()
    iterators = [iter(iterable) for iterable in iterables]
    while True:
        items = [next(iterator, sentinel) for iterator in iterators]
        if all(item is sentinel for item in items):
            return
        if any(item is sentinel for item in items):
            raise ValueError("zip() argument lengths do not match")
        yield tuple(items)
