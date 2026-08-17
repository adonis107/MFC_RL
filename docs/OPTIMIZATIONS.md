# Implementation Optimizations And Result Impact

This note records implementation choices that differ from the most obvious
direct implementation. The purpose is to make clear which changes are only for
runtime/memory/reproducibility, and which ones intentionally change the
estimator, oracle, or experimental regime.

Legend:

- **No result change**: algebraically equivalent to the direct implementation,
  up to floating-point roundoff and possibly a different random-number order.
- **Distributionally unchanged**: samples may differ for the same seed, but the
  estimator samples from the same target distribution.
- **Changes finite-sample behavior**: same asymptotic target or same benchmark
  objective, but finite-batch estimates/training trajectories can differ.
- **Deliberate different mode**: this is an oracle/comparator/study option, not
  the same model-free estimator.

## Shared Algorithm Optimizations

| Optimization | Where | Direct implementation | Why it was made | Result impact |
| --- | --- | --- | --- | --- |
| Flattened parameter vectors plus `format_gradient` | `simplex_mfreinforce.py`, `logits_mfreinforce.py`, `continuous_mfreinforce.py`, `experiments/core/gradient_steps.py` | Keep separate tensor/module parameter structures in every estimator | Gives every estimator one common vector gradient representation, making diagnostics, covariance, MSE, and checkpointing simple | **No result change**. Only reshapes gradients back to the control format |
| Batched trajectory simulation | finite algorithms, continuous algorithm, environments | Simulate each trajectory in a Python loop | Reduces Python overhead and lets PyTorch operate on whole batches | **Distributionally unchanged**. Same stochastic model; seeded sample paths may differ because draws are grouped |
| Weighted score aggregation | `weighted_policy_score_sums` methods | Materialize every per-sample score and multiply/sum later | Avoids storing huge score matrices and computes grouped sums directly | **No result change**, except floating-point summation order |
| Chunked score computation | `_score_chunk_size`, policy score methods | Compute all score vectors at once or one sample at a time | Keeps memory bounded for large neural policies while preserving vectorized AD inside each chunk | **No result change**, except floating-point summation order |
| Batched vector-Jacobian products via `is_grads_batched=True` | neural policy score methods | Run one autograd call per sample/group | Converts many per-sample gradient computations into fewer batched AD calls | **No result change**, except floating-point roundoff |
| Optional score diagnostics | `keep_score_diagnostics` | Always store per-sample score vectors | Per-sample scores are expensive and only needed for diagnostics | **No training result change** when disabled; only diagnostic files are smaller |
| Batch-mean return baselines | simplex, continuous MF-REINFORCE | Use raw returns in the score-function estimator | Reduces gradient variance substantially | **Changes finite-sample behavior**. With the same batch used to estimate the baseline, there can be a small finite-batch bias; asymptotically it vanishes. Use `baseline=null` for the raw estimator |
| Nonzero neural-policy initialization | `experiments/core/controls.py` | Initialize pathwise neural policies through an all-zero controller | Zeroing every hidden weight can freeze hidden-layer learning in Tanh policies because only the final bias receives gradient at initialization | **Changes initialization only**. The objective and estimator are unchanged, but the optimizer starts from a trainable point |
| Classical REINFORCE baseline mode | `experiments/core/reinforce.py` | Compare only MF-REINFORCE/oracle modes | Provides a no-mean-field-score comparator for every benchmark family | **Deliberate different mode**. It omits the population-law score terms by design |

## Discrete MF-REINFORCE Optimizations

| Optimization | Where | Direct implementation | Why it was made | Result impact |
| --- | --- | --- | --- | --- |
| Batched simplex perturbations `sample_q_batch` | `SimplexPerturbedMFREINFORCE` | Sample one simplex perturbation at each path/time in loops | Removes Python loops and lets all perturbations share tensor kernels | **Distributionally unchanged** |
| Closed-form simplex score `H(q)` | `SimplexPerturbedMFREINFORCE.H` | Differentiate the logistic-normal/simplex sampling density through autograd or finite differences | The score has a stable analytic expression and avoids expensive AD through sampling | **No result change** relative to the affine/simplex score being estimated |
| Estimate all state-coordinate sensitivities from one auxiliary batch | `SimplexPerturbedMFREINFORCE.estimate_sensitivity` | Run separate auxiliary simulations for every state coordinate and parameter | Reuses the same auxiliary paths with state-group weights | **No result change** for the estimator formula, except Monte Carlo/RNG ordering |
| Grouped weighted score sums in sensitivity recursion | simplex/logits algorithms and finite environments | Compute full policy score arrays, then select states/groups | Avoids repeated score materialization for each state group | **No result change**, except floating-point summation order |
| Separate outer perturbation `lambda` and sensitivity perturbation `eta` | simplex algorithms | Force `eta = lambda` everywhere | Allows the experiments to study the two perturbation scales separately | **Can change finite-sample behavior**. Same family of estimators, but different `eta` changes bias/variance of the sensitivity estimate |
| `flow_mode = exact` for finite population flows | `runner.finite_population_flow` | Always estimate the mean-field flow by particles | Exact finite-state recursion is cheap and gives cleaner diagnostics/oracle experiments | **Deliberate different mode**. It removes particle-flow Monte Carlo noise and uses model knowledge. Use `flow_mode=particle` for the fully particle-estimated path |
| Exact finite evaluation | `runner.evaluate_finite` | Evaluate trained policies by Monte Carlo rollouts | Gives deterministic mean-field evaluation and cleaner comparisons | **No change to the learned policy**, but reported metrics are exact mean-field metrics rather than rollout estimates |
| Batched logit perturbation estimator | `LogitsPerturbedMFREINFORCE` | Loop over outer samples and auxiliary paths separately | Shares population-flow, perturbation, and score work across samples | **Distributionally unchanged** |
| Sample chunking in logits estimator | `LogitsPerturbedMFREINFORCE.gradient_estimate` | Evaluate all outer samples in one huge tensor | Prevents memory spikes for neural policies | **No result change**, except summation order |
| Adaptive lambda controller | `adaptive_simplex_mfreinforce.py` | Use one fixed perturbation scale | Tries to keep a reasonable bias/variance balance during training | **Deliberate algorithmic change**. It changes the training path and selected perturbation scale |
| Common-random-number diagnostic inside adaptive controller | `checkpoint_diagnostic` | Compare `lambda` and contracted `lambda` with independent random draws | Reduces noise in the cross-scale discrepancy signal | **Changes diagnostic variance**, not the target objective. It can change the adaptive controller's finite-sample decisions |
| Logistic transform for adaptive lambda | `_rho_from_lambda`, `_lambda_from_rho` | Clip lambda directly after every update | Keeps lambda inside valid bounds smoothly | **No direct estimator change**, but it affects controller dynamics |

## Continuous MF-REINFORCE Optimizations

| Optimization | Where | Direct implementation | Why it was made | Result impact |
| --- | --- | --- | --- | --- |
| Low-dimensional coordinate/signature flow | `ContinuousTransportMFREINFORCE.coordinate_dim`, `estimate_coordinate_flow` | Represent the whole random empirical law at every time | The theory uses functionals/signatures of the law; most benchmarks only need a low-dimensional signature | **No change relative to the intended signature-based algorithm**. It is not equivalent to a generic full-law perturbation if the chosen signature is insufficient |
| Exact coordinate flow for LQ and portfolio | `estimate_coordinate_flow` | Estimate the mean law coordinate with particles | These benchmarks have exact moment recursions, so particle noise is unnecessary | **Deliberate benchmark optimization**. It reduces Monte Carlo noise using model knowledge |
| Empirical particle coordinate flow for Cucker-Smale | `estimate_coordinate_flow` | Store and perturb a full measure object | Keeps the nominal empirical law but summarizes the coordinate as mean position/velocity | **Approximation depends on the chosen coordinate chart**. For the current benchmark diagnostics this is the intended chart |
| Fourier-coordinate flow for Kuramoto | `estimate_coordinate_flow`, Kuramoto environment | Use all pairwise phase interactions or a full phase density | The Kuramoto interaction is determined by the first sine/cosine moments | **No model change** for sinusoidal Kuramoto interactions; only roundoff differences |
| Forward cumulative sensitivity score | `ContinuousTransportMFREINFORCE.estimate_sensitivity` | Recompute score prefixes from scratch at every time | Updates the cumulative score once per time step, reducing repeated work | **No result change** for the same paths and perturbations |
| Reuse one auxiliary sensitivity flow in the main batch | `complete_gradient_estimate` | Re-estimate sensitivity inside every main trajectory | Separates auxiliary sensitivity estimation from the outer gradient batch | **Same estimator structure** as the planned algorithm; finite-sample randomness differs from a fully nested implementation |
| Store stage records and compute returns-to-go once | `gradient_estimate` | Recompute trajectories/scores for every time-return pair | Avoids duplicate transition and reward work | **No result change**, except floating-point order |
| Weighted continuous policy score sums | `_weighted_policy_score_sums` | Build every per-sample neural score vector first | Reduces memory and AD calls | **No result change**, except roundoff |
| Separate seeds for nominal flow, sensitivity, and main gradient | `complete_gradient_estimate` | Reuse the same seed stream for all sub-estimators | Avoids accidental coupling between sub-estimators while preserving reproducibility | **Distributionally unchanged**, but exact seeded trajectories differ |
| Benchmark-specific law reconstruction from coordinates | `_law_from_coordinates` | Keep an abstract random measure object | Makes the transport perturbation executable for each environment | **No change for the implemented chart**; results are tied to that chart |
| Oracle coordinate sensitivity for LQ comparator | `continuous_mfreinforce.py`, `experiments/core/gradient_steps.py` | Estimate the mean-flow sensitivity with auxiliary trajectories | Isolates the outer continuous MF-REINFORCE score estimator from sensitivity-estimation noise | **Deliberate oracle/comparator mode**. It changes the estimator by replacing the auxiliary sensitivity estimator with the exact Jacobian |
| Stronger LQ mean-field benchmark config | `experiments/notebooks/configs.py` | Use weak coupling `c=0.1`, `gamma=0.4`, `gamma_T=0.6` | Makes LQ test the mean-field correction rather than mostly the state-feedback term | **Deliberate benchmark change**. It changes the environment used by generated benchmark bundles |

## Environment-Level Optimizations

| Optimization | Where | Direct implementation | Why it was made | Result impact |
| --- | --- | --- | --- | --- |
| Exact finite-state population recursions | finite environments | Estimate population flow by finite-agent simulation | Finite-state mean-field kernels are cheap and differentiable/evaluable | **Deliberate exact/oracle mode** when used for training or diagnostics; evaluation becomes cleaner |
| Tensorized transition tensors and averaged kernels | finite environments | Loop over states/actions manually | Faster and less error-prone population recursion | **No result change** |
| Ring shift for distribution planning | `DistributionPlanningMFC.next_law` | Build a full transition matrix for deterministic ring moves | Uses the ring structure directly | **No result change** |
| Closed-form advertising next-law update | `AdvertisingMFC.next_law` | Sample individual adoption events to estimate the next population law | The mean-field adoption update is available in closed form | **No result change** for the mean-field model |
| Matrix exponential for cybersecurity transitions | `CybersecurityMFC.transition_tensor` | Simulate continuous-time infection/recovery substeps | Computes the exact one-step Markov transition over `dt` | **Changes only the numerical discretization choice**: it is exact for the generator model, not an approximation by substeps |
| Analytic tensor-policy scores for two-state and LQ/portfolio | `policy_score_batch` methods | Use autograd for simple closed-form Gaussian/Bernoulli scores | Faster and avoids unnecessary graph construction | **No result change** |
| Hand-coded MLP weighted score backprop for discrete neural policies | `_neural_policy_scores.py`, advertising/cybersecurity/distribution-planning | Autograd every sample/group separately | Much faster grouped score sums for the fixed Tanh MLP architecture | **No result change for the current architecture**. If the architecture changes, this helper must be updated or bypassed |
| Exact LQ moment recursion and Riccati policy | `LinearQuadraticMFC` | Estimate objective/optimizer by Monte Carlo | Provides oracle objective, gradient, and optimum for validation | **Deliberate oracle/benchmark path**, not model-free |
| Portfolio adjoint exact gradient | `MeanVariancePortfolioMFC.exact_gradient` | Backpropagate through simulated trajectories | Gives exact mean-variance gradient without rollout noise | **Deliberate oracle/benchmark path**, not model-free |
| Standardized Student-t sampler for integer df | `MeanVariancePortfolioMFC._sample_standardized_student_t`, continuous algorithm | Use `torch.distributions.StudentT` always | Preserves explicit generator control for reproducible samples when df is integer | **Distributionally unchanged** for integer df |
| Vectorized Cucker-Smale particle simulator | `CuckerSmaleMFC._simulate_particles` | Loop over particles one by one | Computes pairwise interaction, rewards, and diagnostics in tensor operations | **No result change**, except RNG ordering/roundoff |
| Cucker-Smale pathwise gradient | `CuckerSmaleMFC.pathwise_gradient` | Use score-function estimator only | Gives a lower-variance differentiable-model comparator | **Deliberate different mode**. Requires model differentiability |
| Shared initial particles in Cucker-Smale heuristic grid search | `grid_search_alignment_controller` | Resample initial particles for each heuristic parameter | Makes objective comparisons across heuristic parameters less noisy | **Changes comparison variance**, not the expected objective |
| Lifted phases for Kuramoto dynamics | `KuramotoMFC._simulate_particles` | Wrap phases after every operation and differentiate through wrapped state | Keeps gradients continuous while reporting wrapped physical phases | **No change to circular observables** based on sine/cosine; improves differentiability |
| Fourier moments for Kuramoto interaction and diagnostics | `KuramotoMFC.order_stats`, `interaction_field` | Compute all pairwise `sin(theta_j - theta_i)` terms | For first-harmonic Kuramoto, the first Fourier moment gives the same interaction | **No result change**, except roundoff |
| Kuramoto rejection sampler for von Mises initial phases | `_sample_von_mises` | Depend on an external distribution helper | Keeps dependencies minimal and supports generator-controlled sampling | **Distributionally unchanged** for the intended von Mises law |
| Kuramoto pathwise gradient | `KuramotoMFC.pathwise_gradient` | Use score-function estimator only | Gives a differentiable-model comparator | **Deliberate different mode** |

## Experiment Runner And Artifact Optimizations

| Optimization | Where | Direct implementation | Why it was made | Result impact |
| --- | --- | --- | --- | --- |
| Environment and algorithm registries | `experiments/core/registry.py`, `experiments/runner.py` | Hard-code separate scripts for each benchmark | Gives one CLI and one checkpoint format for all environments | **No result change** |
| Compatibility validation | `validate_compatibility` | Let invalid env/algorithm pairs fail deep inside training | Fails early with clear errors | **No result change** |
| JSON config plus dotted CLI overrides | `apply_overrides` | Maintain many nearly identical scripts | Enables reproducible sweeps without new code | **No result change** when overrides match the intended config |
| Checkpoints store registry names/config/state, not live Python objects | `run_train` | Pickle complete objects | More robust reconstruction and smaller artifacts | **No result change** if code/config are compatible |
| Exact-gradient and pathwise-gradient algorithm names | `experiments/core/registry.py`, `experiments/core/gradient_steps.py` | Only expose MF-REINFORCE algorithms | Makes oracle/comparator runs available through the same pipeline | **Deliberate different mode**, not equivalent to MF-REINFORCE |
| Study runners write tabular CSV summaries | `experiments/studies/` | Generate figures directly inside training scripts | Decouples computation from plotting/notebooks | **No result change** |
| Application diagnostics write notebook-ready CSVs | `experiments/applications/` | Plot application diagnostics directly during evaluation | Makes each benchmark export reusable artifact files for notebooks and tables | **No result change** |
| Notebook helpers regenerate missing bundles lazily | `experiments/notebooks/bundles.py`, `notebook_helpers.py` | Require manual pre-running every command | Makes benchmark notebooks self-contained for smoke or mid-scale checks | **No result change to stored artifacts**, but `QUICK=True` bundles are smoke-test scale, not final paper-scale results |
| Plotting helpers are separated from data generation | `experiments/notebooks/plotting/` | Mix plotting, bundle generation, and CSV loading in one helper file | Keeps computation artifacts independent from visualization code | **No computational result change** |
| Result catalog and coverage matrix | `plot_specs.py`, `experiments/notebooks/coverage.py` | Keep figure requirements only in prose | Tracks which figure families have artifacts and which need more studies | **No computational result change** |

## Choices That Can Affect Numerical Results

These are useful, but they should not be described as purely runtime
optimizations:

1. `flow_mode=exact` versus `flow_mode=particle` for finite-state algorithms.
   Exact mode uses model knowledge and removes particle-flow noise.
2. `exact-gradient`, `pathwise-gradient`, and `continuous-oracle-sensitivity` are oracle/comparator modes, not
   MF-REINFORCE estimators.
3. Adaptive simplex algorithms change `lambda`, sometimes `eta`, and sometimes
   sample sizes over training.
4. Batch-mean baselines reduce variance but can alter finite-batch estimates.
5. The continuous algorithm is signature/coordinate based. Results are for the
   selected `Gamma` chart; underspecified signatures would change the problem.
6. Notebook smoke bundles use tiny horizons/batches by default. They validate
   plumbing and figure generation, not final numerical claims. Use `preset=mid`
   for a stronger laptop-scale repo check before launching `preset=main`.

## Validation Performed

The implementation has smoke and regression coverage for the main optimized
paths:

- invalid env/algorithm compatibility checks;
- nested CLI override handling;
- checkpoint reconstruction from registry/config/state;
- finite and continuous train/diagnostic smoke commands;
- weighted policy score sums against explicit score matrices;
- exact LQ/portfolio gradients against autograd or analytic checks;
- continuous Cucker-Smale/Kuramoto pathwise and score-shape checks;
- notebook helper plot smoke for representative discrete and continuous
  bundles.

Last checked with:

```bash
env PYTHONPATH=src uv run pytest -q
```
