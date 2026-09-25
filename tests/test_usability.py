"""User-facing behaviour of optimize() and the results: clear errors, real parameters, mode inference."""

import numpy as np
import pytest

import pyiea


def _two(g):
    return float(g.sum()), float((1 - g).sum())


@pytest.mark.parametrize(
    ("fitness", "kwargs", "hint"),
    [
        (lambda g: (1.0, 2.0), {}, "use IMOEA"),
        (lambda g: 1.0, {"n_objectives": 2}, "one value per objective"),
        (lambda g: (1.0, 2.0, 3.0), {"n_objectives": 2}, "n_objectives=3"),
    ],
)
def test_wrong_number_of_objectives_is_a_clear_error(fitness, kwargs, hint):
    with pytest.raises(ValueError, match=hint):
        pyiea.optimize(fitness, n_bits=6, max_calls=50, **kwargs)


def test_fitness_failing_everywhere_fails_fast_with_the_first_error():
    calls = []

    def broken(g):
        calls.append(1)
        return 1 / 0

    with pytest.raises(pyiea.EvaluationError, match="ZeroDivisionError") as err:
        pyiea.optimize(broken, n_bits=8, max_calls=1000)
    assert "30 genomes" in str(err.value) and len(calls) == 30  # stopped after the initial population


def test_writing_into_a_genome_explains_read_only():
    def mutating(g):
        g[0] = 1
        return 0.0

    with pytest.raises(pyiea.EvaluationError, match=r"read-only.*g\.copy\(\)"):
        pyiea.optimize(mutating, n_bits=8, max_calls=100)


def test_imoea_fails_fast_too_and_partial_failures_are_fine():
    with pytest.raises(pyiea.EvaluationError, match="initial population"):
        pyiea.optimize(lambda g: (float("nan"), 0.0), n_bits=8, n_objectives=2, max_calls=200)
    # some failures are ordinary: they are counted, and the run goes on
    res = pyiea.optimize(lambda g: 1 / 0 if g[0] else float(g.sum()), n_bits=8, max_calls=200)
    assert res.best == 0.0 and res.accounting["failed_evaluations"] > 0
    # no call at all (zero budget) is not a failure
    assert pyiea.optimize(lambda g: 1 / 0, n_bits=8, max_calls=0).best is None


def test_real_parameters_in_one_call():
    res = pyiea.optimize(lambda x: float(((x - 1.5) ** 2).sum()), bounds=[(-5, 5)] * 3, bits=12, max_calls=3000)
    assert res.best_x is not None and res.best_x.shape == (3,)
    assert np.allclose(res.best_x, 1.5, atol=0.01)
    assert res.best == pytest.approx(float(((res.best_x - 1.5) ** 2).sum()))
    assert "best_x=" in repr(res)
    binary = pyiea.optimize(lambda g: float(g.sum()), n_bits=4, max_calls=50)
    assert binary.best_x is None


def test_real_parameters_multi_objective_front():
    res = pyiea.optimize(lambda x: (float(x[0]), float((1 - x[0]) ** 2 + x[1] ** 2)), bounds=[(0, 1), (-1, 1)],
                         n_objectives=2, max_calls=1500)  # fmt: skip
    X, F = res.front_x, res.front_objectives
    assert X is not None and X.shape == (len(res.front), 2) and F.shape == (len(res.front), 2)
    assert np.allclose(F[:, 0], X[:, 0])  # rows of front_x and front_objectives describe the same points
    assert res.archive_objectives.shape[1] == 2


def test_mode_follows_n_objectives_and_config():
    assert isinstance(pyiea.optimize(_two, n_bits=6, n_objectives=2, max_calls=100), pyiea.IMOEAResult)
    assert isinstance(pyiea.optimize(_two, n_bits=6, config=pyiea.IMOEAConfig(), max_calls=100), pyiea.IMOEAResult)
    assert isinstance(pyiea.optimize(_two, n_bits=6, mode="multi_objective", max_calls=100), pyiea.IMOEAResult)
    assert isinstance(pyiea.optimize(lambda g: float(g.sum()), n_bits=6, max_calls=100), pyiea.IEAResult)
    with pytest.raises(ValueError, match="means one objective"):
        pyiea.optimize(_two, n_bits=6, mode="single_objective", n_objectives=2)
    with pytest.raises(ValueError, match="n_objectives >= 2"):
        pyiea.optimize(_two, n_bits=6, mode="multi_objective", n_objectives=1)
    with pytest.raises(ValueError, match="unknown mode"):
        pyiea.optimize(_two, n_bits=6, mode="multi")
    with pytest.raises(TypeError, match="needs an IEAConfig"):
        pyiea.optimize(lambda g: 0.0, n_bits=6, config=pyiea.IMOEAConfig(), n_objectives=1)


def test_search_space_is_described_once():
    with pytest.raises(ValueError, match="only one of"):
        pyiea.optimize(lambda x: 0.0, n_bits=4, bounds=[(0, 1)])
    with pytest.raises(ValueError, match="does not match"):
        pyiea.optimize(lambda g: 0.0, n_bits=5, problem=pyiea.BinaryProblem(4))
    with pytest.raises(ValueError, match="describe the search space"):
        pyiea.optimize(lambda g: 0.0)
    assert pyiea.optimize(lambda g: float(g.sum()), n_bits=4, problem=pyiea.BinaryProblem(4), max_calls=20).best == 0


def test_results_print_compactly():
    res = pyiea.optimize(_two, n_bits=10, n_objectives=2, max_calls=300)
    text = repr(res)
    assert text.startswith("IMOEAResult(front=") and "points" in text and "array(" not in text
    assert res.front_objectives.tolist() == [list(y) for _, y in res.front]
    single = repr(pyiea.optimize(lambda g: float(g.sum()), n_bits=10, max_calls=100))
    assert single.startswith("IEAResult(best=0,") and "objective_calls=" in single
