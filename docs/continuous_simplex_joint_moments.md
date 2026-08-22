# Continuous Simplex With Mean And Variance

This note records the implementation target for the continuous-state
simplex/MF-REINFORCE estimator. It follows `files/Research_Project.tex`,
section `Continuous State Space`, especially:

- the affine perturbation of a Gaussian population law,
- the density and score of the generated pair `(M_t, Sigma_t)`,
- the likelihood-ratio gradient formula, and
- the model-free moment-sensitivity identities.

It deliberately ignores `files/reference/continuous_state_space(2).tex`.

## Generated Law

At time `t`, the nominal population law is represented as

```text
X_t^theta ~ N(m_t^theta, sigma_t^theta^2).
```

The affine perturbation is

```text
T_t^lambda(x) = (1 + lambda A_t) x + lambda B_t,
```

with independent perturbation coordinates `(A_t, B_t)`. The generated
population argument is therefore

```text
M_t     = (1 + lambda A_t) m_t + lambda B_t,
Sigma_t = (1 + lambda A_t)^2 sigma_t^2.
```

The implementation samples `A_t` and `B_t` as independent centered Gaussians
with variance `rho^2`, matching the benchmark code that produced the closed
forms. The density formula in the TeX assumes `1 + lambda A_t > 0`; the
experiment grids keep this event overwhelmingly likely for the tested
settings, and the code uses the positive-branch score from the original
algorithm.

## Joint Perturbation Score

For a generated point `(M, Sigma)`, write

```text
c = sqrt(Sigma) / sigma = 1 + lambda A,
a = (c - 1) / lambda,
b = (M - m c) / lambda.
```

The joint density from `Research_Project.tex` is

```text
q_t^{lambda,theta}(M, Sigma)
  = f_{A,B}(a, b) / (2 lambda^2 sigma sqrt(Sigma)).
```

For independent Gaussian perturbations,

```text
partial_a log f = -a / rho^2,
partial_b log f = -b / rho^2.
```

Differentiating the log density at fixed `(M, Sigma)` gives

```text
grad_theta log q_t(M, Sigma)
  = h_m(t) grad_theta m_t + h_s(t) grad_theta log sigma_t,

h_m = b c / (lambda rho^2),
h_s = c (a - m b) / (lambda rho^2) - 1.
```

The current codebase parameterizes theta as shape `(T, 2)`, so both
sensitivities have shape `(T+1, T, 2)`.

## Moment Sensitivities

The estimator needs

```text
D_t = grad_theta m_t,
K_t = grad_theta log sigma_t = grad_theta Var(X_t) / (2 Var(X_t)).
```

The model-free auxiliary recursion uses the same likelihood-ratio structure
as the gradient estimator. For a reusable auxiliary batch, maintain a
cumulative score `C_t` up to, but not including, time `t`. Then estimate

```text
D_t     = E[(X_t - m_t) C_t],
V_t'    = E[((X_t - m_t)^2 - sigma_t^2) C_t],
K_t     = V_t' / (2 sigma_t^2).
```

The centered statistics remove deterministic offsets without changing the
expectation because the cumulative score is centered.

After forming `(D_t, K_t)`, update the cumulative score for the next time:

```text
C_{t+1} = C_t
        + h_m(t) D_t
        + h_s(t) K_t
        + policy_score_t.
```

This is the same single-batch forward organization already used by the
mean-only implementation, now with the variance/log-standard-deviation
component included.

## Algorithm Skeleton

One gradient step at parameter `theta`:

1. Draw one auxiliary batch of `n_aux` eta-perturbed trajectories.
2. Along that batch, estimate the moment-sensitivity flow:
   `D_t = grad m_t` and `K_t = grad log sigma_t`.
3. Draw an independent main batch of `B` lambda-perturbed trajectories.
4. For each main trajectory, compute returns-to-go `G_t`.
5. For each time `t`, compute the joint generated-law score
   `h_m(t) D_t + h_s(t) K_t`.
6. Add the direct Gaussian policy score for `t<T`.
7. Average the score-weighted returns over the main batch.

When `K_t` is set to zero, this reduces to the previous mean-only
specialization. The full implementation should use both components by
default.
