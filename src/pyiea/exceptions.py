"""Exceptions raised by pyiea."""

from __future__ import annotations

__all__ = ["BudgetExhaustedError", "CheckpointError", "PyIEAError"]


class PyIEAError(Exception):
    """Base class for every error raised by pyiea."""


class BudgetExhaustedError(PyIEAError):
    """A batch needs more objective calls than the remaining budget allows.

    Raised by :meth:`pyiea.Evaluator.evaluate_batch` before any call is made.
    """


class CheckpointError(PyIEAError):
    """A checkpoint does not belong to this optimizer, configuration, seed or evaluator context."""
