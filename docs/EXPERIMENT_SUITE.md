# MFC Experiment Suite

This project now has a reusable experiment runner for training, diagnostics,
study sweeps, and notebook-ready result exports. The main idea is:

1. Run experiments from JSON configs or CLI overrides.
2. Save every run into a self-contained directory.
3. Build notebooks per environment by reading the saved CSV/JSON artifacts.

The CLI entrypoint is:

```bash
prime-run uv run python main.py <command> --env <environment> --algorithm <algorithm> [options]
```

If the package is not installed, run commands from the repository root.

## Code Organization

The public CLI and notebook APIs are intentionally stable, but the experiment
implementation is split into focused packages:

- `src/mfc/experiments/core/`: registries, config normalization, artifacts,
  checkpoint/control handling, shared runtime helpers, and train-step logic.
- `src/mfc/experiments/diagnostics/`: perturbation, functional-law, score,
  gradient, and sensitivity diagnostics.
- `src/mfc/experiments/applications/`: application-diagnostic artifact writers,
  split by finite-state, exact continuous, pathwise continuous, and reference
  outputs.
- `src/mfc/experiments/studies/`: study dispatch, score validation, grids,
  budget/horizon/scaling studies, robustness, bias studies, adaptive lambda,
  signature ablations, and particle transfer.
- `src/mfc/experiments/notebooks/`: notebook config builders, bundle generation,
  artifact readers, figure coverage tables, and plotting helpers.

The compatibility wrappers `application.py`, `notebook_helpers.py`,
`notebooks/plots.py`, and `studies/__init__.py` remain so older notebooks and
scripts keep working. See `docs/REPO_STRUCTURE.md` for the detailed tree.

## Environments And Algorithms

Environment names:

- `twostate`
- `advertising`
- `cybersecurity`
- `distribution-planning`
- `lq`
- `portfolio`
- `cucker-smale`
- `kuramoto`

Algorithm names:

- `simplex`
- `logits`
- `finite-adaptive-simplex`
- `consistent-adaptive-simplex`
- `continuous-mfreinforce`
- `exact-gradient`
- `pathwise-gradient`

Compatibility rules:

- Finite-state environments use `simplex`, `logits`, and adaptive simplex variants.
- Continuous-state environments can use `continuous-mfreinforce`.
- `lq` and `portfolio` also support `exact-gradient` oracle training/diagnostics.
- `cucker-smale` and `kuramoto` also support `pathwise-gradient` training/diagnostics.

## Run Directory Layout

Each command writes a run directory under `runs/` by default:

```text
runs/<timestamp>_<command>_<env>_<algorithm>_seed<seed>/
  config.json
  metadata.json
  metrics.json
  history.csv              # training only
  checkpoint.pt            # training only
  diagnostics.csv          # diagnostics/studies when applicable
  ... environment-specific CSV files
```

Training checkpoints store registry names, configs, optimizer state, RNG state,
history, metrics, and the learned tensor/module state. They avoid pickling whole
environment or algorithm objects directly.

## JSON Config Shape

Configs are JSON-first:

The default device is `cuda`. On CPU-only machines, pass `--device cpu` or set
`env_config.device` to `"cpu"`.

```json
{
  "env": "twostate",
  "algorithm": "simplex",
  "env_config": {
    "device": "cuda",
    "dtype": "float64",
    "T": 2
  },
  "algorithm_config": {
    "lambda": 0.1,
    "eta": 0.1
  },
  "train": {
    "output_dir": "runs",
    "run_name": "twostate_simplex_demo",
    "seed": 0,
    "steps": 10,
    "lr": 0.001,
    "B": 64,
    "n": 8,
    "validate_every": 1,
    "flow_mode": "exact"
  },
  "evaluation": {
    "horizon": 2,
    "mu0": [0.2, 0.8]
  },
  "diagnostic": {
    "lambdas": [0.03, 0.1, 0.3],
    "replications": 16,
    "samples": 256
  },
  "study": {
    "name": "core-suite"
  }
}
```

Any field can be overridden with `--set dotted.path=json_value`:

```bash
prime-run uv run python main.py train \
  --config configs/twostate.json \
  --set train.steps=100 \
  --set algorithm_config.lambda=0.2 \
  --set evaluation.mu0=[0.2,0.8]
```

Strings in `--set` need JSON quotes if the shell would otherwise pass them as
bare words:

```bash
--set study.name=\"budget-allocation\"
```

## Training

Train and save one model:

```bash
prime-run uv run python main.py train \
  --env twostate \
  --algorithm simplex \
  --output-dir runs \
  --run-name twostate_simplex_smoke \
  --set env_config.device=\"cuda\" \
  --set env_config.dtype=\"float64\" \
  --set env_config.T=2 \
  --set train.steps=5 \
  --set train.B=16 \
  --set train.n=4 \
  --set train.mu0=[0.2,0.8] \
  --set evaluation.mu0=[0.2,0.8] \
  --set algorithm_config.lambda=0.1 \
  --set algorithm_config.eta=0.1
```

For finite-state comparisons, run the same config with `simplex` and `logits`.
The important files for notebooks are:

- `history.csv`: objective/value over training.
- `metrics.json`: final summary.
- `checkpoint.pt`: learned policy/control.

Continuous-state MF-REINFORCE uses the same artifact format:

```bash
prime-run uv run python main.py train \
  --env lq \
  --algorithm continuous-mfreinforce \
  --output-dir runs \
  --run-name lq_continuous_mfreinforce_smoke \
  --set env_config.device=\"cuda\" \
  --set env_config.dtype=\"float64\" \
  --set env_config.T=2 \
  --set train.steps=5 \
  --set train.B=16 \
  --set train.n=4 \
  --set algorithm_config.lambda=0.1 \
  --set algorithm_config.eta=0.1
```

For particle continuous benchmarks, add a nominal coordinate population size:

```bash
--set algorithm_config.population_particles=64 --set train.population_particles=64
```

## Universal Diagnostics

Perturbation geometry:

```bash
prime-run uv run python main.py diagnose-perturbation --config experiment.json
```

Functional-law validation for `Gamma(M^lambda)`:

```bash
prime-run uv run python main.py diagnose-functional-law --config experiment.json
```

Gradient estimator validation:

```bash
prime-run uv run python main.py diagnose-gradient --config experiment.json
```

Sensitivity diagnostics:

```bash
prime-run uv run python main.py diagnose-sensitivity --config experiment.json
```

Population-law score validation:

```bash
prime-run uv run python main.py score-validation --config experiment.json
```

These commands write `diagnostics.csv` plus any extra command-specific CSVs.
They are meant to feed paper plots such as bias/variance/MSE vs lambda,
sensitivity error vs time, score variance vs lambda, and functional-law
covariance summaries.

## Application Diagnostics

Use `application-diagnostics` to export environment-specific datasets:

```bash
prime-run uv run python main.py application-diagnostics \
  --config runs/twostate_simplex_smoke/config.json \
  --set checkpoint=\"runs/twostate_simplex_smoke/checkpoint.pt\"
```

The files depend on the environment:

- Finite-state environments: `population_flow.csv`, `time_metrics.csv`, `policy.csv`.
- LQ/Portfolio: `time_metrics.csv`, `policy.csv`, `terminal_samples.csv`.
- Cucker-Smale: `time_metrics.csv`, `snapshots.csv`.
- Kuramoto: `time_metrics.csv`, `snapshots.csv`.

These are the inputs for per-environment notebooks.

## Studies

The `study` command runs larger experiment families:

```bash
prime-run uv run python main.py study --config experiment.json --set study.name=\"core-suite\"
```

Supported study names:

- `core-suite`: perturbation, functional-law, score when available, sensitivity when available, gradient, application diagnostics.
- `score-validation`: population-law score summaries.
- `horizon-scaling`: gradient diagnostics across `env_config.T`.
- `budget-allocation`: gradient diagnostics across `B:n`.
- `particle-approximation`: diagnostics across empirical population size.
- `scaling`: generic parameter grid.
- `ablation`: component ablations.
- `robustness`: evaluate a checkpoint under changed environment/evaluation settings.
- `optimization-summary`: merge histories/metrics from saved training runs.
- `benchmark-properties`: table of benchmark capabilities.
- `hyperparameters`: flattened reproduction table.
- `application-diagnostics`: same as the direct command.
- `perturbation-bias`: objective/gradient perturbation-bias data where available.
- `optimizer-bias`: retrain across lambda and compare policies/trajectories.
- `signature-ablation`: compare full, reduced, underspecified, or fixed-size signatures.
- `adaptive-lambda`: compare adaptive lambda variants with fixed-lambda baselines.
- `particle-transfer`: train/evaluate particle-count transfer matrices for pathwise benchmarks.

Example budget allocation:

```json
{
  "env": "twostate",
  "algorithm": "simplex",
  "study": {
    "name": "budget-allocation",
    "command": "diagnose-gradient",
    "budgets": [
      {"label": "small", "train.B": 32, "train.n": 4},
      {"label": "medium", "train.B": 64, "train.n": 8}
    ]
  }
}
```

## Result Catalog

Create a catalog mapping paper figures/tables to the commands and files that
produce their data:

```bash
prime-run uv run python main.py catalog --output-dir runs --run-name result_catalog
```

This writes:

- `catalog.json`
- `catalog.csv`
- `metadata.json`
- `metrics.json`

Use this as the checklist when building notebooks.

## Extended Bundle Launcher

The benchmark notebooks default to `EXTENDED = True` and `QUICK = True`, so they
can generate the extended study artifacts needed by `docs/figures.md` without a
paper-scale launch. The same workflow is available from the terminal:

```bash
prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --output-dir runs/extended_benchmark_bundles
```

The launcher has named presets:

- `--preset smoke`: one seed, tiny batches, tiny training, and the same study
  structure as the main run.
- `--preset mid`: one seed, moderate batches, moderate diagnostics, reduced
  oracle/reference settings, and the full study structure. This is intended as
  a laptop-scale full-repo check, roughly around 30 minutes on a CUDA laptop
  such as an NVIDIA 2060 Max-Q, though exact time depends on thermals and
  background load.
- `--preset main`: five seeds, paper defaults, and the lambda/eta/horizon grids
  below.
- `--preset high-confidence`: ten seeds and larger diagnostic replications for
  the few headline estimator figures.

Use `--full` as an alias for `--preset main`, `--force` to rebuild existing
artifacts, `--no-progress` for quieter logs, and `--env twostate --env lq` to
run only selected benchmarks. The
gap checker reports any coverage rows that are still marked `future` or
`study-needed`:

```bash
prime-run uv run python scripts/check_experiment_gap.py --fail-on-missing
```

Main preset defaults:

```text
main seeds: [0, 1, 2, 3, 4]
adaptive lambda seeds: [0, 1, 2, 3, 4]
high-confidence seeds: [0, ..., 9]

mid lambda/eta grid:
  [0.025, 0.1, 0.2]

lambda/eta grid:
  [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]

small-lambda asymptotic fit subset:
  [0.0125, 0.025, 0.05, 0.1]

main diagnostic replications/samples:
  128 / 2048

high-confidence diagnostic replications/samples:
  512 / 4096
```

Mid training defaults:

```text
twostate:              steps=100, B=32, n=4, horizons=[2, 8]
advertising:           steps=80,  B=32, n=4, horizons=[3, 10]
cybersecurity:         steps=80,  B=32, n=4, horizons=[2, 5]
distribution-planning: steps=80,  B=32, n=4, horizons=[3, 10]
lq:                    steps=100, B=64, n=8, horizons=[3, 10]
portfolio:             steps=100, B=64, n=8, horizons=[3, 10]
cucker-smale:          steps=40,  B=24, n=4, N_pop=32, N_val=128, horizons=[10, 20]
kuramoto:              steps=40,  B=24, n=4, N_pop=32, N_val=128, horizons=[10, 20]

mid diagnostic replications/samples:
  8 / 96

mid budget grid:
  B=[16, 32], n=[2, 4]

mid particle/signature grids:
  particles=[16, 32, 64], signature_dims=[1, 2, 4]
```

Main training defaults:

```text
twostate:              steps=10_000, B=256, n=16, horizons=[2, 8, 32]
advertising:           steps=25_000, B=256, n=16, horizons=[5, 20, 40]
cybersecurity:         steps=25_000, B=256, n=16, horizons=[3, 10, 20]
distribution-planning: steps=30_000, B=256, n=16, horizons=[5, 20, 40]
lq:                    steps=10_000, B=512, n=32, horizons=[5, 20, 40]
portfolio:             steps=15_000, B=512, n=32, horizons=[5, 20, 40]
cucker-smale:          steps=30_000, B=256, n=32, N_pop=512, N_val=4096, horizons=[25, 50, 100]
kuramoto:              steps=40_000, B=256, n=32, N_pop=512, N_val=4096, horizons=[50, 100, 200]
```

Smoke training defaults keep the same experiment structure with one seed,
`lambda/eta=[0.05, 0.1, 0.8]`, `diagnostic.replications=4`, and
`diagnostic.samples=32`.

## Notebook Workflow

Recommended workflow per environment:

1. Run training jobs for the algorithms to compare.
2. Run `application-diagnostics` on each saved checkpoint.
3. Run universal diagnostics or studies for the estimator/theory plots.
4. Build the notebook from CSV/JSON only.

For example, a two-state notebook can read:

- `history.csv` from simplex and logits training runs.
- `population_flow.csv`, `time_metrics.csv`, and `policy.csv` from application diagnostics.
- `diagnostics.csv` from gradient, sensitivity, score, and perturbation studies.
- `catalog.csv` to keep figure coverage explicit.

Current benchmark notebooks:

- `notebooks/twostate_benchmark.ipynb`
- `notebooks/advertising_benchmark.ipynb`
- `notebooks/cybersecurity_benchmark.ipynb`
- `notebooks/distribution_planning_benchmark.ipynb`
- `notebooks/lq_benchmark.ipynb`
- `notebooks/portfolio_benchmark.ipynb`
- `notebooks/cucker_smale_benchmark.ipynb`
- `notebooks/kuramoto_benchmark.ipynb`

This keeps long training separate from visualization and makes notebooks fast to
rerun.
