# Complete Run Readiness

## Status

The experiment suite is ready to run CUDA-first training, diagnostics, and
paper-scale study grids from the registry/CLI/notebook helpers. The missing
items from the earlier readiness audit now have artifact writers or study
drivers. The remaining work is to choose final budgets/seeds, run the matrix on
a machine where CUDA is visible, and inspect the generated notebooks.

The current suite can already produce the shared artifact format for:

- training runs and checkpoints;
- perturbation geometry summaries;
- functional-law summary diagnostics;
- score validation summaries;
- gradient bias/variance/MSE summaries;
- sensitivity summaries;
- horizon, budget, particle, ablation, robustness, and optimization-summary
  study scaffolds;
- benchmark-specific application diagnostics;
- per-benchmark notebooks that load saved artifacts.

Before a complete paper run, the remaining work is operational: final run
selection, GPU availability, and enough wall-clock budget.

## Current Module Layout

The experiment code has been split into maintenance-oriented packages while
preserving the public imports used by scripts and notebooks:

- `experiments/core/`: registry, checkpoint, artifact, runtime, control, and
  training-step helpers.
- `experiments/diagnostics/`: perturbation, functional-law, gradient,
  sensitivity, and shared diagnostic code.
- `experiments/applications/`: notebook-ready application artifact writers and
  benchmark reference outputs.
- `experiments/studies/`: study dispatcher plus score, grid, report, bias, and
  advanced study families.
- `experiments/notebooks/`: config builders, bundle generation, artifact
  loading, coverage tables, and plotting.

The wrappers `application.py`, `notebook_helpers.py`, `notebooks/plots.py`, and
`studies/__init__.py` are intentionally small compatibility layers.

## Device Default

The default device is now `cuda` in the runner, environment configs, notebook
helpers, and examples. On CPU-only machines, override with `--device cpu` or
`--set env_config.device=\"cpu\"`.

Checkpoint loading still uses CPU-safe deserialization by default; reconstructed
controls are moved through the saved config/device path afterward.

## Implemented Support

1. Raw functional-law samples.

   `diagnose-functional-law` now writes `signature_samples.csv`,
   `functional_covariance.csv`, and `functional_distances.csv` in addition to
   `diagnostics.csv`.

2. Coordinatewise gradient artifacts.

   `diagnose-gradient` now writes `gradient_samples.csv`,
   `gradient_coordinates.csv`, and `gradient_covariance.csv`, including oracle
   coordinates, coordinate bias/MSE, confidence intervals, coverage, sign
   accuracy, and covariance entries when an oracle is available.

3. Optimizer-bias experiments across lambda.

   `study.name="optimizer-bias"` retrains at each lambda and writes
   `diagnostics.csv` plus `optimization_history.csv` with control distance,
   policy-output distance, trajectory distance, and unperturbed performance
   relative to a reference run.

4. Sensitivity-method comparisons.

   `diagnose-sensitivity` accepts `diagnostic.methods`, including
   `auxiliary`, `reused_main`, `oracle`, `finite_difference`, and
   `pathwise_ad` where the environment exposes the required deterministic
   sensitivity. It writes `sensitivity_samples.csv`.

5. Horizon scaling.

   `study.name="horizon-scaling"` can run gradient, sensitivity, and score
   components over a horizon grid and records runtime, simulator-budget proxy,
   and peak CUDA memory.

6. Budget allocation.

   `study.name="budget-allocation"` supports `B_values`, `n_values`, explicit
   budget variants, runtime, simulator-budget proxy, and peak CUDA memory.

7. Variance-reduction ablations.

   `algorithm_config.antithetic=true` is implemented for simplex, logit, and
   continuous coordinate perturbations. `study.name="ablation"` includes
   baseline, no-baseline, flow-mode, antithetic, and sensitivity-baseline
   variants where compatible.

8. Signature sufficiency and dimension ablations.

   `diagnostic.signature_mode`, `diagnostic.signature_dim`, and
   `diagnostic.signature_coordinates` are supported. `study.name` can be set to
   `signature-ablation`.

9. Adaptive-lambda studies.

   `study.name="adaptive-lambda"` runs fixed-lambda baselines and finite-state
   adaptive algorithms, and writes `lambda_trace.csv` when child histories
   include lambda values.

10. Robustness/generalization sweeps.

    `study.name="robustness"` now has benchmark-aware default shifts for
    initial laws, horizons, costs, coupling strengths, noise levels, Student-t
    return distributions, infection pressure, and advertising parameters.

11. Particle approximation and transfer matrices.

    `study.name="particle-approximation"` handles particle sweeps, and
    `study.name="particle-transfer"` writes trained-at/evaluated-at particle
    matrices for continuous particle benchmarks.

12. Scaling experiments.

    `study.name="scaling"` has a default grid over horizon, batch sizes, and
    particles where relevant, with runtime, simulator-budget proxy, peak CUDA
    memory, and child diagnostics.

13. Application-specific discrete artifacts.

    - Two-state: `landscape.csv` includes objective landscape and gradient
      field.
    - Distribution planning: `transport_flux.csv` records ring transport flux.
    - Advertising: `finite_population.csv` records finite-population versus
      mean-field adoption gaps.
    - Cybersecurity robustness variants are available through the robustness
      study defaults.

14. Application-specific continuous artifacts.

    - LQ: `landscape.csv` contains objective slices around the current policy.
    - Portfolio: `efficient_frontier.csv` contains oracle-interpolation frontier
      points.
    - Cucker-Smale and Kuramoto: `post_control.csv` records post-control free
      evolution.
    - Particle transfer and robustness variants are available through their
      study drivers.

15. Master tables.

    Benchmark properties, hyperparameters, optimization summaries, and the
    catalog all point to the expanded artifact set.

16. Gap-analysis scripts and notebook extended mode.

    Each benchmark notebook now has `EXTENDED = True`, so quick notebook runs
    generate the extended study artifacts. The terminal equivalents are
    `prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset smoke`
    and `prime-run uv run python scripts/check_experiment_gap.py`.

    For a deeper laptop-scale check before the paper run, use
    `prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset mid`.

    For the main paper run, use
    `prime-run uv run python scripts/run_extended_benchmark_suite.py --env all --preset main`.

17. GPU smoke.

    Use `prime-run uv run python ...` on this project. Rerun the CUDA smoke
    before launching a complete CUDA run.

18. Documentation and repository organization.

    The current module layout is documented in `docs/REPO_STRUCTURE.md`.
    Legacy two-state experiment scripts, old notebooks, and old result images
    have been moved under `archive/` so active source paths only contain the
    modular suite.

## Remaining Before Final Paper Runs

1. Ensure `prime-run` exposes CUDA on the target machine.
2. Run the full study matrix once with the `mid` preset and inspect failures or
   pathological figures.
3. Run the full study matrix with the `main` preset.
4. Execute the benchmark notebooks against the final run directories.
5. Inspect figures for statistical stability and rerun weak grids with larger
   budgets where confidence intervals are too wide.
6. Commit the refactor and generated notebook/docs changes in coherent chunks
   after the final smoke run.
