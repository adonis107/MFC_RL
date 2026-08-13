# MFC RL Experiment Suite

This repository contains finite-state and continuous-state mean-field-control
reinforcement-learning experiments, with reusable training, diagnostics, and
notebook workflows.

## Layout

- `src/mfc/environments/`: benchmark environments.
- `src/mfc/algorithms/`: simplex, logits, adaptive, exact/pathwise, and continuous MF-REINFORCE algorithms.
- `src/mfc/experiments/`: modular experiment runner, diagnostics, studies, application artifact writers, and notebook helpers.
- `notebooks/`: one benchmark notebook per environment plus the experiment gap analysis.
- `scripts/`: suite-level commands for smoke/full benchmark generation and coverage checks.
- `docs/`: experiment-suite guide, repository structure, figure map, optimization notes, and complete-run readiness notes.
- `reference/`: source TeX reference material used to align implementation and experiments.
- `runs/`, `models/`, `results/`, `archive/`, `files/`: ignored local artifacts/reference archives.

## Quick Checks

Run the test suite:

```bash
env PYTHONPATH=src uv run pytest -q
```

Run a GPU-backed smoke bundle:

```bash
prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset smoke --core-only
```

Run a laptop-scale full-repo check:

```bash
prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset mid
```

The suite launcher prints a per-env/per-seed progress list by default. Use
`--no-progress` for quieter batch logs.

Run a full paper preset:

```bash
prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset main
```

Use the modular CLI directly:

```bash
prime-run uv run python main.py train --config path/to/config.json
prime-run uv run python main.py diagnose-gradient --config path/to/config.json
```

Generated outputs are written under `runs/` by default and are intentionally not tracked.

For the detailed module layout, see `docs/REPO_STRUCTURE.md`.
