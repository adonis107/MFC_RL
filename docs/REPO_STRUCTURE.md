# Repository Structure

This repository is organized around stable public entry points and smaller
implementation packages. Existing notebooks and scripts should keep importing
the public wrappers; the subpackages are for maintenance and extension.

## Public Entry Points

- `main.py`: command-line entry point for training, diagnostics, studies, and
  result catalogs.
- `scripts/run_extended_benchmark_suite.py`: smoke/mid/main/high-confidence
  bundle launcher for all benchmark environments.
- `scripts/check_experiment_gap.py`: reports figure families still marked as
  missing or future work.
- `mfc.experiments.notebook_helpers`: notebook-facing compatibility facade.
- `mfc.experiments.application.run_application_diagnostics`: stable application
  diagnostics import.
- `mfc.experiments.studies`: stable study-runner import surface.

## Experiment Package

```text
src/mfc/experiments/
  cli.py                    # argparse command implementations used by main.py
  runner.py                 # train command wrapper and legacy public runner API
  plot_specs.py             # result catalog / figure-family metadata
  presets.py                # smoke/mid/main/high-confidence budgets and grids

  core/
    registry.py             # environment and algorithm registries
    session.py              # config normalization, seeding, checkpoint loading
    artifacts.py            # run directories, JSON/CSV writers, overrides
    controls.py             # control initialization, serialization, vectors
    gradient_steps.py       # algorithm construction and train gradient steps
    evaluation.py           # shared policy evaluation helpers
    runtime.py              # batch, horizon, lambda, validation-law helpers

  diagnostics/
    perturbation.py         # perturbation geometry diagnostics
    functional_law.py       # Gamma(M^lambda) diagnostics
    gradient.py             # gradient bias/variance/MSE diagnostics
    sensitivity.py          # sensitivity diagnostics
    common.py               # shared diagnostic helpers

  applications/
    runner.py               # run_application_diagnostics
    finite.py               # finite-state application artifacts
    references.py           # finite-state oracle/reference outputs
    exact.py                # LQ/portfolio exact application artifacts
    pathwise.py             # Cucker-Smale/Kuramoto application artifacts
    common.py               # shared row builders and checkpoint loading

  studies/
    dispatch.py             # run_study dispatcher
    core_suite.py           # core-suite orchestration
    score.py                # population-law score validation
    grids.py                # parameter grids, budget, horizon, scaling, ablation
    reports.py              # robustness, summaries, benchmark/hyperparameter tables
    bias.py                 # perturbation and optimizer bias studies
    advanced.py             # adaptive lambda, signature ablation, particle transfer
    common.py               # shared study utilities

  notebooks/
    configs.py              # benchmark config builders
    bundles.py              # benchmark bundle generation
    data.py                 # CSV/JSON readers
    coverage.py             # figure checklist and gap tables
    plots.py                # compatibility wrapper
    plotting/
      training.py           # training curves
      discrete.py           # finite-state notebook plots
      continuous.py         # continuous-state notebook plots
      diagnostics.py        # universal diagnostic plots
      studies.py            # study-summary plots
      display.py            # display_all_figures helpers
      common.py             # shared plotting helpers
```

## Compatibility Wrappers

These files intentionally remain small wrappers:

- `src/mfc/experiments/application.py`
- `src/mfc/experiments/notebook_helpers.py`
- `src/mfc/experiments/notebooks/plots.py`
- `src/mfc/experiments/studies/__init__.py`

They keep older imports working while the implementation lives in the packages
listed above.

## Artifact Directories

- `runs/`: generated training, diagnostic, study, and catalog outputs.
- `notebooks/`: benchmark notebooks that read saved artifacts.
- `archive/`: old notebooks, old scripts, and older result images kept for
  reference.

Generated artifacts are intentionally ignored by git unless explicitly moved
into a tracked documentation or reference location.
