# Contributing

## Setup

```bash
git clone https://github.com/Leowwd/pyiea.git
cd pyiea
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest --cov
```

CI runs the same checks on Python 3.10–3.13.

## Rules

- **Paper fidelity.** Any change to the algorithm's behaviour must update `docs/paper_map.md`. Mark the change as *faithful*, *eng* (an engineering choice where the paper is silent), or *variant*. Do not call a modified algorithm IEA or IMOEA; call it an "IEA-based variant" and list the differences.
- **Accounting.** Every objective call goes through `Evaluator`. Do not call a fitness function directly from the algorithms.
- **Immutability.** Genomes are read-only (`freeze`). Never modify a parent in place.
- **Tests.** Test new logic against a result you can check independently, such as an enumerated optimum, a hand-computed example, or a value printed in the paper. Tests must be deterministic: seed every random generator.
- **Commits.** Use [Conventional Commits](https://www.conventionalcommits.org/), for example `feat(igc): ...`, `fix(evaluator): ...` or `docs: ...`.
- **Versioning.** Follow [SemVer](https://semver.org/), and add changes under `## [Unreleased]` in `CHANGELOG.md`.

## Releasing

Releases go to PyPI through `.github/workflows/release.yml`, triggered by a tag `vX.Y.Z`. It re-runs the checks, builds the sdist and wheel, checks that the tag matches `project.version` and that `CHANGELOG.md` has a `## [X.Y.Z]` section, runs the test suite against the built wheel, and publishes with [trusted publishing](https://docs.pypi.org/trusted-publishers/). No API token is stored.

One-time setup (maintainer):

1. On PyPI, add a pending trusted publisher for the project `pyiea`: owner `Leowwd`, repository `pyiea`, workflow `release.yml`, environment `pypi`.
2. On GitHub, create the environment `pypi` (Settings → Environments). Adding yourself as a required reviewer makes every upload wait for your approval.

Each release:

1. Set `version` in `pyproject.toml`. Move the `[Unreleased]` entries of `CHANGELOG.md` into `## [X.Y.Z] - YYYY-MM-DD`.
2. Once the package is on PyPI, replace the README's "not on PyPI yet" install section with `pip install pyiea`.
3. Merge to `main`, then tag that commit and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
