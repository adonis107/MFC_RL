from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional

import torch


Controller = Callable[[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


@dataclass
class KuramotoConfig:
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float64

    T: int = 100
    dt: float = 0.05
    K: float = 0.3
    D: float = 0.2
    theta_star: float = 0.0

    kappa0: float = 10.0
    kappa_sync: float = 1.0
    kappa_lock: float = 0.25
    kappa_sync_T: float = 5.0
    kappa_lock_T: float = 2.0
    gamma: float = 0.05

    tau: float = 0.1
    a_max: float = 3.0
    rho: float = 1.0
    sigma_omega: float = 0.0
    include_frequency: bool = False

    hidden_units: int = 64
    N_pop: int = 1000
    N_val: int = 5000

    lr: float = 1e-3
    n_train: int = 100_000
    training_runs: int = 5
    validate_every: int = 10
    keep_score_diagnostics: bool = False

    @property
    def uses_intrinsic_frequencies(self) -> bool:
        return self.sigma_omega > 0.0

    @property
    def feature_dim(self) -> int:
        return 10 if self.include_frequency else 9

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.K <= 0.0:
            raise ValueError("K must be positive.")
        if self.D < 0.0:
            raise ValueError("D must be non-negative.")
        if self.kappa0 < 0.0:
            raise ValueError("kappa0 must be non-negative.")
        if self.kappa_sync < 0.0 or self.kappa_lock < 0.0:
            raise ValueError("running cost weights must be non-negative.")
        if self.kappa_sync_T < 0.0 or self.kappa_lock_T < 0.0:
            raise ValueError("terminal cost weights must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive.")
        if self.rho < 0.0:
            raise ValueError("rho must be non-negative.")
        if self.sigma_omega < 0.0:
            raise ValueError("sigma_omega must be non-negative.")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive.")
        if self.N_pop <= 0 or self.N_val <= 0:
            raise ValueError("N_pop and N_val must be positive.")


class KuramotoPolicy(torch.nn.Module):
    def __init__(self, config: KuramotoConfig):
        super().__init__()
        self.config = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.feature_dim, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, 1),
        ).to(device=config.device, dtype=config.dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=self.config.dtype, device=self.config.device)
        return self.config.a_max * torch.tanh(self.net(features).squeeze(-1))


class KuramotoMFC:
    """
    Controlled noisy Kuramoto benchmark on the circle.

    Rollouts keep lifted phases for differentiability and expose wrapped phases
    for physical diagnostics. Population quantities are computed from the first
    empirical Fourier moment, avoiding pairwise particle sums.
    """

    def __init__(self, config: KuramotoConfig):
        self.config = config
        self.state_dim = 1
        self.action_dim = 1

    def zero_policy(self) -> KuramotoPolicy:
        policy = KuramotoPolicy(self.config)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        return policy

    @property
    def critical_coupling(self) -> float:
        return 2.0 * self.config.D

    def wrap_phases(self, phases: torch.Tensor) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        return torch.remainder(phases, 2.0 * math.pi)

    def order_stats(self, phases: torch.Tensor) -> Dict[str, torch.Tensor]:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        c = torch.cos(phases).mean(dim=-1)
        s = torch.sin(phases).mean(dim=-1)
        r = torch.sqrt(c.square() + s.square())
        aligned = math.cos(self.config.theta_star) * c + math.sin(self.config.theta_star) * s
        return {"C": c, "S": s, "R": r, "target_aligned": aligned}

    def order_parameter(self, phases: torch.Tensor) -> torch.Tensor:
        return self.order_stats(phases)["R"]

    def target_aligned_order(self, phases: torch.Tensor) -> torch.Tensor:
        return self.order_stats(phases)["target_aligned"]

    def interaction_field(self, phases: torch.Tensor, law_phases: torch.Tensor) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        stats = self.order_stats(law_phases)
        c = self._broadcast_stat(stats["C"], phases)
        s = self._broadcast_stat(stats["S"], phases)
        return s * torch.cos(phases) - c * torch.sin(phases)

    def local_disagreement(self, phases: torch.Tensor, law_phases: torch.Tensor) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        stats = self.order_stats(law_phases)
        c = self._broadcast_stat(stats["C"], phases)
        s = self._broadcast_stat(stats["S"], phases)
        return 1.0 - torch.cos(phases) * c - torch.sin(phases) * s

    def target_locking_cost(self, phases: torch.Tensor) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        return 1.0 - torch.cos(phases - self.config.theta_star)

    def cross_circular_dispersion(self, phases: torch.Tensor, law_phases: torch.Tensor) -> torch.Tensor:
        phase_stats = self.order_stats(phases)
        law_stats = self.order_stats(law_phases)
        return 1.0 - phase_stats["C"] * law_stats["C"] - phase_stats["S"] * law_stats["S"]

    def order_evolution_terms(
        self,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        interaction = self.interaction_field(phases, law_phases)
        r = self.order_parameter(phases)
        return {
            "endogenous": 2.0 * self.config.K * interaction.square().mean(dim=-1),
            "control": 2.0 * (interaction * actions).mean(dim=-1),
            "diffusion": -2.0 * self.config.D * r.square(),
        }

    def policy_features(
        self,
        t: int,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        stats = self.order_stats(law_phases)
        c = self._broadcast_stat(stats["C"], phases)
        s = self._broadcast_stat(stats["S"], phases)
        r = self._broadcast_stat(stats["R"], phases)
        interaction = self.interaction_field(phases, law_phases)
        time = torch.full_like(phases, float(t) / max(1, self.config.T))
        features = [
            time,
            torch.cos(phases),
            torch.sin(phases),
            c,
            s,
            interaction,
            r,
            torch.sin(self.config.theta_star - phases),
            torch.cos(self.config.theta_star - phases),
        ]
        if self.config.include_frequency:
            if frequencies is None:
                omega = torch.zeros_like(phases)
            else:
                omega = torch.as_tensor(frequencies, dtype=phases.dtype, device=phases.device)
                omega = torch.broadcast_to(omega, phases.shape)
            features.append(omega)
        return torch.stack(features, dim=-1)

    def policy_mean(
        self,
        policy: KuramotoPolicy,
        t: int,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return policy(self.policy_features(t, phases, law_phases, frequencies))

    def free_action_mean(
        self,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        return torch.zeros_like(phases)

    def free_controller(self) -> Controller:
        return lambda t, phases, law_phases, frequencies=None: self.free_action_mean(phases, law_phases, frequencies)

    def base_action_mean(
        self,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        kappa: float,
        nu: float,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        return float(kappa) * self.interaction_field(phases, law_phases) + float(nu) * torch.sin(
            self.config.theta_star - phases
        )

    def base_controller(self, kappa: float, nu: float) -> Controller:
        return lambda t, phases, law_phases, frequencies=None: self.base_action_mean(
            phases,
            law_phases,
            kappa,
            nu,
            frequencies,
        )

    def running_cost_batch(
        self,
        phases: torch.Tensor,
        actions: torch.Tensor,
        law_phases: torch.Tensor,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=phases.dtype, device=phases.device)
        return self.config.dt * (
            self.config.kappa_sync * self.local_disagreement(phases, law_phases)
            + self.config.kappa_lock * self.target_locking_cost(phases)
            + self.config.gamma * actions.square()
        )

    def terminal_cost_batch(self, phases: torch.Tensor, law_phases: torch.Tensor) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        return (
            self.config.kappa_sync_T * self.local_disagreement(phases, law_phases)
            + self.config.kappa_lock_T * self.target_locking_cost(phases)
        )

    def running_reward_batch(
        self,
        phases: torch.Tensor,
        actions: torch.Tensor,
        law_phases: torch.Tensor,
    ) -> torch.Tensor:
        return -self.running_cost_batch(phases, actions, law_phases)

    def terminal_reward_batch(self, phases: torch.Tensor, law_phases: torch.Tensor) -> torch.Tensor:
        return -self.terminal_cost_batch(phases, law_phases)

    def policy_log_probs_batch(
        self,
        policy: KuramotoPolicy,
        t: int,
        law_phases: torch.Tensor,
        phases: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor] = None,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=phases.dtype, device=phases.device)
        if action_means is None:
            action_means = self.policy_mean(policy, t, phases, law_phases, frequencies)
        else:
            action_means = torch.as_tensor(action_means, dtype=phases.dtype, device=phases.device)
        residuals = actions - action_means
        tau_sq = self.config.tau**2
        return -0.5 * (residuals.square() / tau_sq + math.log(2.0 * math.pi * tau_sq))

    def policy_scores_batch(
        self,
        policy: KuramotoPolicy,
        t: int,
        law_phases: torch.Tensor,
        phases: torch.Tensor,
        actions: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = tuple(policy.parameters())
        phases_flat = phases.reshape(-1)
        actions_flat = actions.reshape(-1)
        frequencies_flat = None if frequencies is None else torch.as_tensor(
            frequencies,
            dtype=self.config.dtype,
            device=self.config.device,
        ).reshape(-1)
        if chunk_size is None:
            chunk_size = min(64, phases_flat.shape[0])

        chunks = []
        for start in range(0, phases_flat.shape[0], chunk_size):
            end = min(start + chunk_size, phases_flat.shape[0])
            frequency_chunk = None if frequencies_flat is None else frequencies_flat[start:end]
            logp = self.policy_log_probs_batch(
                policy,
                t,
                law_phases,
                phases_flat[start:end],
                actions_flat[start:end],
                frequencies=frequency_chunk,
            )
            grad_outputs = torch.eye(end - start, dtype=self.config.dtype, device=self.config.device)
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                allow_unused=False,
            )
            chunks.append(torch.cat([g.reshape(end - start, -1) for g in grads], dim=1).detach())
        return torch.cat(chunks, dim=0).reshape(*phases.shape, -1)

    def policy_score_batch(
        self,
        policy: KuramotoPolicy,
        t: int,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor] = None,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.policy_scores_batch(policy, t, law_phases, phases, actions, frequencies=frequencies)

    def weighted_policy_score_sums(
        self,
        policy: KuramotoPolicy,
        t: int,
        law_phases: torch.Tensor,
        phases: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = tuple(policy.parameters())
        phases_flat = phases.reshape(-1)
        actions_flat = actions.reshape(-1)
        frequencies_flat = None if frequencies is None else torch.as_tensor(
            frequencies,
            dtype=self.config.dtype,
            device=self.config.device,
        ).reshape(-1)
        weights_2d = weights.to(dtype=self.config.dtype, device=self.config.device).reshape(-1, phases_flat.shape[0])
        group_shape = weights.shape[:-1] if weights.shape != phases.shape else ()
        param_dim = sum(parameter.numel() for parameter in params)
        result = torch.zeros(weights_2d.shape[0], param_dim, dtype=self.config.dtype, device=self.config.device)
        if chunk_size is None:
            chunk_size = phases_flat.shape[0]

        for start in range(0, phases_flat.shape[0], chunk_size):
            end = min(start + chunk_size, phases_flat.shape[0])
            chunk_weights = weights_2d[:, start:end]
            active = chunk_weights.abs().sum(dim=1) > 0
            if not active.any():
                continue
            frequency_chunk = None if frequencies_flat is None else frequencies_flat[start:end]
            logp = self.policy_log_probs_batch(
                policy,
                t,
                law_phases,
                phases_flat[start:end],
                actions_flat[start:end],
                frequencies=frequency_chunk,
            )
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=chunk_weights[active],
                is_grads_batched=True,
                allow_unused=False,
            )
            flat = torch.cat([g.reshape(chunk_weights[active].shape[0], -1) for g in grads], dim=1)
            result[active] += flat.detach()

        if group_shape:
            return result.reshape(*group_shape, param_dim)
        return result.squeeze(0)

    @torch.no_grad()
    def sample_initial_phases(
        self,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive.")
        generator = self._generator(seed) if generator is None and seed is not None else generator
        signs = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n, dtype=self.config.dtype, device=self.config.device),
            generator=generator,
        ) - 1.0
        centers = signs * (0.5 * math.pi)
        return self._sample_von_mises(centers, self.config.kappa0, generator)

    @torch.no_grad()
    def sample_frequencies(
        self,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive.")
        generator = self._generator(seed) if generator is None and seed is not None else generator
        if self.config.sigma_omega == 0.0:
            return torch.zeros(n, dtype=self.config.dtype, device=self.config.device)
        return self.config.sigma_omega * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)

    @torch.no_grad()
    def sample_actions_batch(
        self,
        policy: KuramotoPolicy,
        t: int,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_mean = self.policy_mean(policy, t, phases, law_phases, frequencies)
        noise = self.config.tau * torch.randn_like(action_mean)
        return action_mean + noise

    @torch.no_grad()
    def sample_next_phases_batch(
        self,
        phases: torch.Tensor,
        actions: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phases = torch.as_tensor(phases, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=phases.dtype, device=phases.device)
        frequency_drift = torch.zeros_like(phases) if frequencies is None else torch.as_tensor(
            frequencies,
            dtype=phases.dtype,
            device=phases.device,
        )
        noise = torch.sqrt(torch.as_tensor(2.0 * self.config.D * self.config.dt, dtype=phases.dtype, device=phases.device))
        noise = noise * torch.randn(phases.shape, dtype=phases.dtype, device=phases.device)
        next_lifted = phases + self.config.dt * (
            frequency_drift + self.config.K * self.interaction_field(phases, law_phases) + actions
        ) + noise
        return self.wrap_phases(next_lifted)

    @torch.no_grad()
    def sample_trajectories(
        self,
        policy: KuramotoPolicy | Controller,
        n: int,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        initial_phases: Optional[torch.Tensor] = None,
        frequencies: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        rollout = self._simulate_particles(
            policy,
            n,
            seed=seed,
            lambda_=lambda_,
            initial_phases=initial_phases,
            frequencies=frequencies,
            horizon=horizon,
            exploration=exploration,
        )
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in rollout.items()}

    def particle_objective(
        self,
        policy: KuramotoPolicy | Controller,
        n_particles: Optional[int] = None,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        initial_phases: Optional[torch.Tensor] = None,
        frequencies: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> torch.Tensor:
        n = self.config.N_pop if n_particles is None else n_particles
        rollout = self._simulate_particles(
            policy,
            n,
            seed=seed,
            lambda_=lambda_,
            initial_phases=initial_phases,
            frequencies=frequencies,
            horizon=horizon,
            exploration=exploration,
        )
        return rollout["objective"]

    def pathwise_gradient(
        self,
        policy: KuramotoPolicy,
        n_particles: Optional[int] = None,
        replications: int = 1,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        horizon: Optional[int] = None,
        exploration: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if replications <= 0:
            raise ValueError("replications must be positive.")
        params = tuple(policy.parameters())
        generator = self._generator(seed)
        n = self.config.N_pop if n_particles is None else n_particles
        objective = torch.zeros((), dtype=self.config.dtype, device=self.config.device)
        for _ in range(replications):
            rollout = self._simulate_particles(
                policy,
                n,
                generator=generator,
                lambda_=lambda_,
                horizon=horizon,
                exploration=exploration,
            )
            objective = objective + rollout["objective"]
        objective = objective / replications
        grads = torch.autograd.grad(objective, params, allow_unused=False)
        grad_flat = torch.cat([grad.reshape(-1) for grad in grads])
        return objective.detach(), grad_flat.detach()

    @torch.no_grad()
    def grid_search_base_controller(
        self,
        kappa_grid: torch.Tensor,
        nu_grid: torch.Tensor,
        n_particles: Optional[int] = None,
        seed: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        kappa_grid = torch.as_tensor(kappa_grid, dtype=self.config.dtype, device=self.config.device)
        nu_grid = torch.as_tensor(nu_grid, dtype=self.config.dtype, device=self.config.device)
        if kappa_grid.ndim != 1 or kappa_grid.numel() == 0:
            raise ValueError("kappa_grid must be a non-empty one-dimensional tensor.")
        if nu_grid.ndim != 1 or nu_grid.numel() == 0:
            raise ValueError("nu_grid must be a non-empty one-dimensional tensor.")
        n = self.config.N_val if n_particles is None else n_particles
        generator = self._generator(seed)
        initial_phases = self.sample_initial_phases(n, generator=generator)
        frequencies = self.sample_frequencies(n, generator=generator)
        objectives = torch.empty(kappa_grid.numel(), nu_grid.numel(), dtype=self.config.dtype, device=self.config.device)
        for i, kappa in enumerate(kappa_grid):
            for j, nu in enumerate(nu_grid):
                controller = self.base_controller(float(kappa.item()), float(nu.item()))
                objectives[i, j] = self.particle_objective(
                    controller,
                    n_particles=n,
                    initial_phases=initial_phases,
                    frequencies=frequencies,
                    horizon=horizon,
                    exploration=False,
                )
        best_flat = objectives.argmin()
        best_i = best_flat // nu_grid.numel()
        best_j = best_flat % nu_grid.numel()
        return {
            "kappa_grid": kappa_grid,
            "nu_grid": nu_grid,
            "objectives": objectives,
            "best_kappa": kappa_grid[best_i],
            "best_nu": nu_grid[best_j],
        }

    @torch.no_grad()
    def continue_uncontrolled(
        self,
        initial_phases: torch.Tensor,
        steps: int,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        frequencies: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        phases = torch.as_tensor(initial_phases, dtype=self.config.dtype, device=self.config.device)
        if phases.ndim != 1:
            raise ValueError("initial_phases must be one-dimensional.")
        return self.sample_trajectories(
            self.free_controller(),
            phases.shape[0],
            seed=seed,
            lambda_=lambda_,
            initial_phases=phases,
            frequencies=frequencies,
            horizon=steps,
            exploration=False,
        )

    def synchronization_time(self, order_parameter: torch.Tensor, threshold: float = 0.9) -> float:
        return self._hitting_time_above(order_parameter, threshold)

    def phase_locking_time(self, target_aligned_order: torch.Tensor, threshold: float = 0.85) -> float:
        return self._hitting_time_above(target_aligned_order, threshold)

    def _simulate_particles(
        self,
        policy: KuramotoPolicy | Controller,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        lambda_: float = 0.0,
        initial_phases: Optional[torch.Tensor] = None,
        frequencies: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive.")
        horizon = self.config.T if horizon is None else horizon
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        lambda_value = float(lambda_)
        if lambda_value < 0.0:
            raise ValueError("lambda_ must be non-negative.")
        generator = self._generator(seed) if generator is None else generator
        use_exploration = self._uses_exploration(policy, exploration)

        if initial_phases is None:
            lifted_t = self._sample_initial_phases_differentiable(n, generator)
        else:
            lifted_t = torch.as_tensor(initial_phases, dtype=self.config.dtype, device=self.config.device)
            if lifted_t.shape != (n,):
                raise ValueError(f"initial_phases must have shape {(n,)}, got {tuple(lifted_t.shape)}.")

        if frequencies is None:
            omega = self._sample_frequencies_differentiable(n, generator)
        else:
            omega = torch.as_tensor(frequencies, dtype=self.config.dtype, device=self.config.device)
            if omega.shape != (n,):
                raise ValueError(f"frequencies must have shape {(n,)}, got {tuple(omega.shape)}.")

        perturbations = self._sample_perturbations(horizon, generator)
        lifted_flow = [lifted_t]
        phase_flow = [self.wrap_phases(lifted_t)]
        perturbed_flow = []
        actions = []
        action_means = []
        residuals = []
        interactions = []
        stage_costs = []
        order_parameters = []
        perturbed_order_parameters = []
        target_aligned_orders = []
        perturbed_target_aligned_orders = []
        synchronization_costs = []
        control_energy = []

        for t in range(horizon):
            law_phases_t = self._perturb_phases(lifted_t, perturbations[t], lambda_value)
            perturbed_flow.append(self.wrap_phases(law_phases_t))
            mean_action = self._action_mean(policy, t, lifted_t, law_phases_t, omega)
            if use_exploration:
                residual = self.config.tau * torch.randn(
                    lifted_t.shape,
                    dtype=self.config.dtype,
                    device=self.config.device,
                    generator=generator,
                )
            else:
                residual = torch.zeros_like(mean_action)
            action = mean_action + residual
            interaction = self.interaction_field(lifted_t, law_phases_t)
            noise_scale = torch.sqrt(
                torch.as_tensor(2.0 * self.config.D * self.config.dt, dtype=self.config.dtype, device=self.config.device)
            )
            noise = noise_scale * torch.randn(
                lifted_t.shape,
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            cost = self.running_cost_batch(lifted_t, action, law_phases_t)

            actions.append(action)
            action_means.append(mean_action)
            residuals.append(residual)
            interactions.append(interaction)
            stage_costs.append(cost)
            order_parameters.append(self.order_parameter(lifted_t))
            perturbed_order_parameters.append(self.order_parameter(law_phases_t))
            target_aligned_orders.append(self.target_aligned_order(lifted_t))
            perturbed_target_aligned_orders.append(self.target_aligned_order(law_phases_t))
            synchronization_costs.append(self.cross_circular_dispersion(lifted_t, law_phases_t))
            control_energy.append(action.square().mean())

            lifted_t = lifted_t + self.config.dt * (omega + self.config.K * interaction + action) + noise
            lifted_flow.append(lifted_t)
            phase_flow.append(self.wrap_phases(lifted_t))

        terminal_law_phases = self._perturb_phases(lifted_t, perturbations[horizon], lambda_value)
        perturbed_flow.append(self.wrap_phases(terminal_law_phases))
        terminal_costs = self.terminal_cost_batch(lifted_t, terminal_law_phases)
        order_parameters.append(self.order_parameter(lifted_t))
        perturbed_order_parameters.append(self.order_parameter(terminal_law_phases))
        target_aligned_orders.append(self.target_aligned_order(lifted_t))
        perturbed_target_aligned_orders.append(self.target_aligned_order(terminal_law_phases))
        synchronization_costs.append(self.cross_circular_dispersion(lifted_t, terminal_law_phases))

        lifted_flow_tensor = torch.stack(lifted_flow)
        phase_flow_tensor = torch.stack(phase_flow)
        perturbed_flow_tensor = torch.stack(perturbed_flow)
        actions_tensor = torch.stack(actions, dim=1)
        action_means_tensor = torch.stack(action_means, dim=1)
        residuals_tensor = torch.stack(residuals, dim=1)
        stage_costs_tensor = torch.stack(stage_costs, dim=1)
        objective = stage_costs_tensor.mean(dim=0).sum() + terminal_costs.mean()

        return {
            "phase_flow": phase_flow_tensor,
            "lifted_phase_flow": lifted_flow_tensor,
            "perturbed_phase_flow": perturbed_flow_tensor,
            "states": phase_flow_tensor.transpose(0, 1),
            "lifted_states": lifted_flow_tensor.transpose(0, 1),
            "perturbed_states": perturbed_flow_tensor.transpose(0, 1),
            "frequencies": omega,
            "actions": actions_tensor,
            "action_means": action_means_tensor,
            "residuals": residuals_tensor,
            "interaction_fields": torch.stack(interactions, dim=1),
            "stage_costs": stage_costs_tensor,
            "terminal_costs": terminal_costs,
            "particle_costs": stage_costs_tensor.sum(dim=1) + terminal_costs,
            "objective": objective,
            "order_parameter": torch.stack(order_parameters),
            "perturbed_order_parameter": torch.stack(perturbed_order_parameters),
            "target_aligned_order": torch.stack(target_aligned_orders),
            "perturbed_target_aligned_order": torch.stack(perturbed_target_aligned_orders),
            "synchronization_cost": torch.stack(synchronization_costs),
            "control_energy": torch.stack(control_energy),
            "cumulative_control_energy": self.config.dt * torch.stack(control_energy).sum(),
            "perturbations": perturbations,
        }

    def _action_mean(
        self,
        policy: KuramotoPolicy | Controller,
        t: int,
        phases: torch.Tensor,
        law_phases: torch.Tensor,
        frequencies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(policy, KuramotoPolicy):
            return self.policy_mean(policy, t, phases, law_phases, frequencies)
        if isinstance(policy, torch.nn.Module):
            return policy(self.policy_features(t, phases, law_phases, frequencies))
        return policy(t, phases, law_phases, frequencies)

    def _uses_exploration(self, policy: KuramotoPolicy | Controller, exploration: Optional[bool]) -> bool:
        if exploration is not None:
            return bool(exploration)
        return isinstance(policy, torch.nn.Module)

    def _sample_initial_phases_differentiable(
        self,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        signs = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n, dtype=self.config.dtype, device=self.config.device),
            generator=generator,
        ) - 1.0
        centers = signs * (0.5 * math.pi)
        return self._sample_von_mises(centers, self.config.kappa0, generator)

    def _sample_frequencies_differentiable(
        self,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.config.sigma_omega == 0.0:
            return torch.zeros(n, dtype=self.config.dtype, device=self.config.device)
        return self.config.sigma_omega * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)

    def _sample_von_mises(
        self,
        centers: torch.Tensor,
        concentration: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        centers = torch.as_tensor(centers, dtype=self.config.dtype, device=self.config.device)
        if concentration <= 1e-12:
            uniform = 2.0 * math.pi * torch.rand(
                centers.shape,
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            return centers + uniform - math.pi

        kappa = float(concentration)
        a = 1.0 + math.sqrt(1.0 + 4.0 * kappa**2)
        b = (a - math.sqrt(2.0 * a)) / (2.0 * kappa)
        r = (1.0 + b**2) / (2.0 * b)
        samples = torch.empty_like(centers)
        remaining = torch.ones(centers.shape, dtype=torch.bool, device=self.config.device)

        while bool(remaining.any()):
            remaining_indices = remaining.nonzero(as_tuple=False).squeeze(-1)
            count = remaining_indices.numel()
            u1 = torch.rand(count, dtype=self.config.dtype, device=self.config.device, generator=generator)
            u2 = torch.rand(count, dtype=self.config.dtype, device=self.config.device, generator=generator)
            u3 = torch.rand(count, dtype=self.config.dtype, device=self.config.device, generator=generator)
            z = torch.cos(math.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            accept = (u2 < c * (2.0 - c)) | (torch.log(c / u2) + 1.0 - c >= 0.0)
            if not bool(accept.any()):
                continue
            accepted_indices = remaining_indices[accept]
            signs = torch.where(u3[accept] > 0.5, 1.0, -1.0)
            angles = signs * torch.acos(f[accept].clamp(-1.0, 1.0))
            samples[accepted_indices] = centers[accepted_indices] + angles
            remaining[accepted_indices] = False

        return samples

    def _sample_perturbations(
        self,
        horizon: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        beta0 = self.config.rho * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        beta_c = self.config.rho * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        beta_s = self.config.rho * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        return torch.stack([beta0, beta_c, beta_s], dim=-1)

    def _perturb_phases(
        self,
        phases: torch.Tensor,
        perturbation: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor:
        if lambda_value == 0.0 or self.config.rho == 0.0:
            return self.wrap_phases(phases)
        beta0, beta_c, beta_s = perturbation.unbind(dim=-1)
        transported = phases + lambda_value * (beta0 + beta_c * torch.cos(phases) + beta_s * torch.sin(phases))
        return self.wrap_phases(transported)

    def _broadcast_stat(self, stat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while stat.ndim < target.ndim:
            stat = stat.unsqueeze(-1)
        return torch.broadcast_to(stat, target.shape)

    def _hitting_time_above(self, values: torch.Tensor, threshold: float) -> float:
        values = torch.as_tensor(values, dtype=self.config.dtype, device=self.config.device)
        threshold_value = float(threshold)
        for t in range(values.numel()):
            if bool((values[t:] >= threshold_value).all()):
                return float(t)
        return float("inf")

    def _generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator = torch.Generator(device=self.config.device)
        generator.manual_seed(int(seed))
        return generator
