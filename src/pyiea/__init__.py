"""pyiea: IEA and IMOEA with the intelligent gene collector (IGC).

Implements Ho, Shu, Chen (2004), "Intelligent Evolutionary Algorithms for Large
Parameter Optimization Problems", IEEE TEVC 8(6):522-541, for binary genomes:

* :class:`IEA`: single-objective mode.
* :class:`IMOEA`: multi-objective mode with GPSIFF fitness and elite sets.
* :func:`optimize`: one-call entry point.

All objectives are minimized.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from .api import optimize
from .checkpoint import load_checkpoint, save_checkpoint
from .encoding import ParameterProblem, RealEncoder
from .evaluator import EvalResult, Evaluator, Objective
from .exceptions import BudgetExhaustedError, CheckpointError, PyIEAError
from .iea import IEA, IEAConfig, IEAResult
from .igc import IGCResult, igc
from .imoea import IMOEA, IMOEAConfig, IMOEAResult
from .oa import generate_oa
from .pareto import ParetoSet, coverage, dominance_matrix, gpsiff, hypervolume_2d, igd, nondominated_mask
from .problem import BinaryProblem, FixedCardinalityProblem, Genome, Problem, freeze

try:
    __version__ = version("pyiea")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "IEA",
    "IMOEA",
    "BinaryProblem",
    "BudgetExhaustedError",
    "CheckpointError",
    "EvalResult",
    "Evaluator",
    "FixedCardinalityProblem",
    "Genome",
    "IEAConfig",
    "IEAResult",
    "IGCResult",
    "IMOEAConfig",
    "IMOEAResult",
    "Objective",
    "ParetoSet",
    "Problem",
    "ParameterProblem",
    "PyIEAError",
    "RealEncoder",
    "__version__",
    "coverage",
    "dominance_matrix",
    "freeze",
    "generate_oa",
    "gpsiff",
    "hypervolume_2d",
    "igc",
    "igd",
    "load_checkpoint",
    "nondominated_mask",
    "optimize",
    "save_checkpoint",
]
