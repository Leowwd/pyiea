"""Checkpoint files.

Checkpoints are pickles. **Never load a checkpoint from an untrusted source**:
unpickling can execute arbitrary code.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

__all__ = ["STOPPING_FIELDS", "algorithm_config", "load_checkpoint", "save_checkpoint"]

STOPPING_FIELDS = frozenset({"max_generations", "target", "max_stall_generations", "checkpoint_every"})
"""Config fields that may change between a checkpoint and its resume (to extend a run)."""


def algorithm_config(config: dict[str, Any]) -> dict[str, Any]:
    """The part of a config that must match for a resume: everything but stopping conditions."""
    return {k: v for k, v in config.items() if k not in STOPPING_FIELDS}


def save_checkpoint(path: str | os.PathLike[str], state: dict[str, Any]) -> None:
    """Write ``state`` atomically (write to a temporary file, then rename)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`. Trusted files only."""
    with open(path, "rb") as f:
        state: dict[str, Any] = pickle.load(f)  # noqa: S301 - documented: trusted files only
    return state
