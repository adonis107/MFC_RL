from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


Controller = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class CuckerSmaleConfig:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    T: int = 50
    dt: float = 0.1
    K: float = 1.0
    beta: float = 1.0
    gamma: float = 0.05
    kappa_T: float = 5.0
    tau: float = 0.1
    a_max: float = 2.0
    sigma: float = 0.0
    rho_x: float = 1.0
    rho_v: float = 1.0
    eps_num: float = 1e-12

    hidden_units: int = 64
    N_pop: int = 1000
    N_val: int = 5000

    cluster_position: float = 2.0
    cluster_velocity: float = 1.0
    position_std: float = 0.2
    velocity_std: float = 0.1

    lr: float = 1e-3
    n_train: int = 100_000
    training_runs: int = 5
    validate_every: int = 10
    keep_score_diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.K <= 0.0:
            raise ValueError("K must be positive.")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if self.kappa_T < 0.0:
            raise ValueError("kappa_T must be non-negative.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        if self.rho_x < 0.0 or self.rho_v < 0.0:
            raise ValueError("rho_x and rho_v must be non-negative.")
        if self.eps_num <= 0.0:
            raise ValueError("eps_num must be positive.")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive.")
        if self.N_pop <= 0 or self.N_val <= 0:
            raise ValueError("N_pop and N_val must be positive.")
        if self.position_std < 0.0 or self.velocity_std < 0.0:
            raise ValueError("position_std and velocity_std must be non-negative.")


class CuckerSmalePolicy(torch.nn.Module):
    def __init__(self, config: CuckerSmaleConfig):
        super().__init__()
        self.config = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(7, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, config.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_units, 1),
        ).to(device=config.device, dtype=config.dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=self.config.dtype, device=self.config.device)
        return self.config.a_max * torch.tanh(self.net(features).squeeze(-1))


class CuckerSmaleMFC:
    """
    One-dimensional controlled Cucker-Smale flocking benchmark.

    The state is (position, velocity). Population quantities are represented by
    an empirical particle law, and the Gaussian neural policy consumes the
    feature vector described in the benchmark specification.
    """

    def __init__(self, config: CuckerSmaleConfig):
        self.config = config
        self.state_dim = 2
        self.action_dim = 1

    def zero_policy(self) -> CuckerSmalePolicy:
        policy = CuckerSmalePolicy(self.config)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        return policy

    def communication_kernel(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.as_tensor(distances, dtype=self.config.dtype, device=self.config.device)
        return self.config.K / (1.0 + distances.square()).pow(self.config.beta)

    def empirical_stats(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        if states.shape[-1] != self.state_dim:
            raise ValueError("states must have final dimension 2.")
        x_mean = states[..., 0].mean(dim=-1)
        v_mean = states[..., 1].mean(dim=-1)
        v_variance = (states[..., 1] - v_mean.unsqueeze(-1)).square().mean(dim=-1)
        return {"x_mean": x_mean, "v_mean": v_mean, "v_variance": v_variance}

    def velocity_dispersion(self, states: torch.Tensor) -> torch.Tensor:
        return self.empirical_stats(states)["v_variance"]

    def spatial_diameter(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        positions = states[..., 0]
        return positions.max(dim=-1).values - positions.min(dim=-1).values

    def alignment_field(self, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_states = torch.as_tensor(law_states, dtype=self.config.dtype, device=self.config.device)
        if states.shape[-1] != self.state_dim or law_states.shape[-1] != self.state_dim:
            raise ValueError("states and law_states must have final dimension 2.")

        x = states[..., 0]
        v = states[..., 1]
        law_x = law_states[..., 0]
        law_v = law_states[..., 1]
        distances = torch.abs(x.unsqueeze(-1) - law_x.unsqueeze(-2))
        weights = self.communication_kernel(distances)
        return (weights * (law_v.unsqueeze(-2) - v.unsqueeze(-1))).mean(dim=-1)

    def _broadcast_stat(self, stat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while stat.ndim < target.ndim:
            stat = stat.unsqueeze(-1)
        return torch.broadcast_to(stat, target.shape)

    def policy_features(self, t: int, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_states = torch.as_tensor(law_states, dtype=self.config.dtype, device=self.config.device)
        stats = self.empirical_stats(law_states)
        x = states[..., 0]
        v = states[..., 1]
        x_mean = self._broadcast_stat(stats["x_mean"], x)
        v_mean = self._broadcast_stat(stats["v_mean"], x)
        v_dispersion = self._broadcast_stat(stats["v_variance"], x)
        alignment = self.alignment_field(states, law_states)
        time = torch.full_like(x, float(t) / max(1, self.config.T))
        return torch.stack(
            [
                time,
                x,
                v,
                x_mean,
                v_mean,
                alignment,
                torch.sqrt(v_dispersion + self.config.eps_num),
            ],
            dim=-1,
        )

    def policy_mean(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        states: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        return policy(self.policy_features(t, states, law_states))

    def free_action_mean(self, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        return torch.zeros(states.shape[:-1], dtype=states.dtype, device=states.device)

    def free_controller(self) -> Controller:
        return lambda t, states, law_states: self.free_action_mean(states, law_states)

    def global_alignment_action_mean(
        self,
        states: torch.Tensor,
        law_states: torch.Tensor,
        kappa: float,
    ) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_states = torch.as_tensor(law_states, dtype=self.config.dtype, device=self.config.device)
        v_mean = self.empirical_stats(law_states)["v_mean"]
        return -float(kappa) * (states[..., 1] - self._broadcast_stat(v_mean, states[..., 1]))

    def global_alignment_controller(self, kappa: float) -> Controller:
        return lambda t, states, law_states: self.global_alignment_action_mean(states, law_states, kappa)

    def running_cost_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=self.config.dtype, device=self.config.device)
        law_states = torch.as_tensor(law_states, dtype=self.config.dtype, device=self.config.device)
        v_mean = self.empirical_stats(law_states)["v_mean"]
        v_error = states[..., 1] - self._broadcast_stat(v_mean, states[..., 1])
        return self.config.dt * (v_error.square() + self.config.gamma * actions.square())

    def terminal_cost_batch(self, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        law_states = torch.as_tensor(law_states, dtype=self.config.dtype, device=self.config.device)
        v_mean = self.empirical_stats(law_states)["v_mean"]
        v_error = states[..., 1] - self._broadcast_stat(v_mean, states[..., 1])
        return self.config.kappa_T * v_error.square()

    def running_reward_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        return -self.running_cost_batch(states, actions, law_states)

    def terminal_reward_batch(self, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        return -self.terminal_cost_batch(states, law_states)

    def policy_log_probs_batch(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        law_states: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=self.config.dtype, device=self.config.device)
        if action_means is None:
            action_means = self.policy_mean(policy, t, states, law_states)
        else:
            action_means = torch.as_tensor(action_means, dtype=states.dtype, device=states.device)
        residuals = actions - action_means
        return -0.5 * residuals.square() / (self.config.tau**2)

    def policy_scores_batch(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        law_states: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = tuple(policy.parameters())
        states_flat = states.reshape(-1, self.state_dim)
        actions_flat = actions.reshape(-1)
        if chunk_size is None:
            chunk_size = min(64, states_flat.shape[0])

        chunks = []
        for start in range(0, states_flat.shape[0], chunk_size):
            end = min(start + chunk_size, states_flat.shape[0])
            logp = self.policy_log_probs_batch(policy, t, law_states, states_flat[start:end], actions_flat[start:end])
            grad_outputs = torch.eye(end - start, dtype=self.config.dtype, device=self.config.device)
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                allow_unused=False,
            )
            chunks.append(torch.cat([g.reshape(end - start, -1) for g in grads], dim=1).detach())
        return torch.cat(chunks, dim=0).reshape(*states.shape[:-1], -1)

    def policy_score_batch(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        states: torch.Tensor,
        law_states: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.policy_scores_batch(policy, t, law_states, states, actions)

    def weighted_policy_score_sums(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        law_states: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = tuple(policy.parameters())
        states_flat = states.reshape(-1, self.state_dim)
        actions_flat = actions.reshape(-1)
        weights_2d = weights.to(dtype=self.config.dtype, device=self.config.device).reshape(-1, states_flat.shape[0])
        group_shape = weights.shape[:-1] if weights.shape != states.shape[:-1] else ()
        param_dim = sum(parameter.numel() for parameter in params)
        result = torch.zeros(weights_2d.shape[0], param_dim, dtype=self.config.dtype, device=self.config.device)
        if chunk_size is None:
            chunk_size = states_flat.shape[0]

        for start in range(0, states_flat.shape[0], chunk_size):
            end = min(start + chunk_size, states_flat.shape[0])
            chunk_weights = weights_2d[:, start:end]
            active = chunk_weights.abs().sum(dim=1) > 0
            if not active.any():
                continue
            logp = self.policy_log_probs_batch(policy, t, law_states, states_flat[start:end], actions_flat[start:end])
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
    def sample_initial_states(
        self,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if n <= 0:
            raise ValueError("n must be positive.")
        generator = self._generator(seed) if generator is None and seed is not None else generator
        c = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n, dtype=self.config.dtype, device=self.config.device),
            generator=generator,
        ) - 1.0
        x = self.config.cluster_position * c
        x = x + self.config.position_std * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
        v = self.config.cluster_velocity * c
        v = v + self.config.velocity_std * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
        return torch.stack([x, v], dim=-1)

    @torch.no_grad()
    def sample_actions_batch(
        self,
        policy: CuckerSmalePolicy,
        t: int,
        states: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        action_mean = self.policy_mean(policy, t, states, law_states)
        noise = self.config.tau * torch.randn_like(action_mean)
        return action_mean + noise

    @torch.no_grad()
    def sample_next_states_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        alignment = self.alignment_field(states, law_states)
        noise = self.config.sigma * (self.config.dt**0.5) * torch.randn(
            states.shape[:-1],
            dtype=self.config.dtype,
            device=self.config.device,
        )
        x_next = states[..., 0] + self.config.dt * states[..., 1]
        v_next = states[..., 1] + self.config.dt * (alignment + actions) + noise
        return torch.stack([x_next, v_next], dim=-1)

    @torch.no_grad()
    def sample_trajectories(
        self,
        policy: CuckerSmalePolicy | Controller,
        n: int,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        initial_states: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        rollout = self._simulate_particles(
            policy,
            n,
            seed=seed,
            lambda_=lambda_,
            initial_states=initial_states,
            horizon=horizon,
            exploration=exploration,
        )
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in rollout.items()}

    def particle_objective(
        self,
        policy: CuckerSmalePolicy | Controller,
        n_particles: Optional[int] = None,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
        initial_states: Optional[torch.Tensor] = None,
        horizon: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> torch.Tensor:
        n = self.config.N_pop if n_particles is None else n_particles
        rollout = self._simulate_particles(
            policy,
            n,
            seed=seed,
            lambda_=lambda_,
            initial_states=initial_states,
            horizon=horizon,
            exploration=exploration,
        )
        return rollout["objective"]

    def pathwise_gradient(
        self,
        policy: CuckerSmalePolicy,
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
    def grid_search_alignment_controller(
        self,
        kappa_grid: torch.Tensor,
        n_particles: Optional[int] = None,
        seed: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        kappa_grid = torch.as_tensor(kappa_grid, dtype=self.config.dtype, device=self.config.device)
        if kappa_grid.ndim != 1 or kappa_grid.numel() == 0:
            raise ValueError("kappa_grid must be a non-empty one-dimensional tensor.")
        n = self.config.N_val if n_particles is None else n_particles
        initial_states = self.sample_initial_states(n, seed=seed)
        objectives = torch.empty_like(kappa_grid)
        for idx, kappa in enumerate(kappa_grid):
            controller = self.global_alignment_controller(float(kappa.item()))
            objectives[idx] = self.particle_objective(
                controller,
                n_particles=n,
                initial_states=initial_states,
                horizon=horizon,
                exploration=False,
            )
        best_idx = objectives.argmin()
        return {"kappa_grid": kappa_grid, "objectives": objectives, "best_kappa": kappa_grid[best_idx]}

    @torch.no_grad()
    def continue_uncontrolled(
        self,
        initial_states: torch.Tensor,
        steps: int,
        seed: Optional[int] = None,
        lambda_: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        states = torch.as_tensor(initial_states, dtype=self.config.dtype, device=self.config.device)
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError("initial_states must have shape (n, 2).")
        return self.sample_trajectories(
            self.free_controller(),
            states.shape[0],
            seed=seed,
            lambda_=lambda_,
            initial_states=states,
            horizon=steps,
            exploration=False,
        )

    def alignment_time(self, velocity_dispersion: torch.Tensor, threshold: float) -> float:
        dispersion = torch.as_tensor(velocity_dispersion, dtype=self.config.dtype, device=self.config.device)
        threshold_value = float(threshold)
        for t in range(dispersion.numel()):
            if bool((dispersion[t:] <= threshold_value).all()):
                return float(t)
        return float("inf")

    def _simulate_particles(
        self,
        policy: CuckerSmalePolicy | Controller,
        n: int,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        lambda_: float = 0.0,
        initial_states: Optional[torch.Tensor] = None,
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

        if initial_states is None:
            states_t = self._sample_initial_states_differentiable(n, generator)
        else:
            states_t = torch.as_tensor(initial_states, dtype=self.config.dtype, device=self.config.device)
            if states_t.shape != (n, self.state_dim):
                raise ValueError(f"initial_states must have shape {(n, self.state_dim)}, got {tuple(states_t.shape)}.")

        perturbations = self._sample_perturbations(horizon, generator)
        state_flow = [states_t]
        perturbed_flow = []
        actions = []
        action_means = []
        residuals = []
        alignments = []
        stage_costs = []
        v_dispersion = []
        perturbed_v_dispersion = []
        control_energy = []

        for t in range(horizon):
            law_states_t = self._perturb_states(states_t, perturbations[t], lambda_value)
            perturbed_flow.append(law_states_t)
            mean_action = self._action_mean(policy, t, states_t, law_states_t)
            if use_exploration:
                residual = self.config.tau * torch.randn(
                    states_t.shape[:-1],
                    dtype=self.config.dtype,
                    device=self.config.device,
                    generator=generator,
                )
            else:
                residual = torch.zeros_like(mean_action)
            action = mean_action + residual
            alignment = self.alignment_field(states_t, law_states_t)
            noise = self.config.sigma * (self.config.dt**0.5) * torch.randn(
                states_t.shape[:-1],
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            cost = self.running_cost_batch(states_t, action, law_states_t)

            actions.append(action)
            action_means.append(mean_action)
            residuals.append(residual)
            alignments.append(alignment)
            stage_costs.append(cost)
            v_dispersion.append(self.velocity_dispersion(states_t))
            perturbed_v_dispersion.append(self.velocity_dispersion(law_states_t))
            control_energy.append(action.square().mean())

            x_next = states_t[..., 0] + self.config.dt * states_t[..., 1]
            v_next = states_t[..., 1] + self.config.dt * (alignment + action) + noise
            states_t = torch.stack([x_next, v_next], dim=-1)
            state_flow.append(states_t)

        terminal_law_states = self._perturb_states(states_t, perturbations[horizon], lambda_value)
        perturbed_flow.append(terminal_law_states)
        terminal_costs = self.terminal_cost_batch(states_t, terminal_law_states)
        v_dispersion.append(self.velocity_dispersion(states_t))
        perturbed_v_dispersion.append(self.velocity_dispersion(terminal_law_states))

        state_flow_tensor = torch.stack(state_flow)
        perturbed_flow_tensor = torch.stack(perturbed_flow)
        actions_tensor = torch.stack(actions, dim=1)
        action_means_tensor = torch.stack(action_means, dim=1)
        residuals_tensor = torch.stack(residuals, dim=1)
        stage_costs_tensor = torch.stack(stage_costs, dim=1)
        objective = stage_costs_tensor.mean(dim=0).sum() + terminal_costs.mean()

        return {
            "state_flow": state_flow_tensor,
            "perturbed_state_flow": perturbed_flow_tensor,
            "states": state_flow_tensor.transpose(0, 1),
            "perturbed_states": perturbed_flow_tensor.transpose(0, 1),
            "actions": actions_tensor,
            "action_means": action_means_tensor,
            "residuals": residuals_tensor,
            "alignment_fields": torch.stack(alignments, dim=1),
            "stage_costs": stage_costs_tensor,
            "terminal_costs": terminal_costs,
            "particle_costs": stage_costs_tensor.sum(dim=1) + terminal_costs,
            "objective": objective,
            "velocity_dispersion": torch.stack(v_dispersion),
            "perturbed_velocity_dispersion": torch.stack(perturbed_v_dispersion),
            "control_energy": torch.stack(control_energy),
            "cumulative_control_energy": self.config.dt * torch.stack(control_energy).sum(),
            "spatial_diameter": self.spatial_diameter(state_flow_tensor),
            "perturbations": perturbations,
        }

    def _action_mean(
        self,
        policy: CuckerSmalePolicy | Controller,
        t: int,
        states: torch.Tensor,
        law_states: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(policy, CuckerSmalePolicy):
            return self.policy_mean(policy, t, states, law_states)
        if isinstance(policy, torch.nn.Module):
            return policy(self.policy_features(t, states, law_states))
        return policy(t, states, law_states)

    def _uses_exploration(self, policy: CuckerSmalePolicy | Controller, exploration: Optional[bool]) -> bool:
        if exploration is not None:
            return bool(exploration)
        return isinstance(policy, torch.nn.Module)

    def _sample_initial_states_differentiable(
        self,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        c = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n, dtype=self.config.dtype, device=self.config.device),
            generator=generator,
        ) - 1.0
        x = self.config.cluster_position * c
        x = x + self.config.position_std * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
        v = self.config.cluster_velocity * c
        v = v + self.config.velocity_std * torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
        return torch.stack([x, v], dim=-1)

    def _sample_perturbations(
        self,
        horizon: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        zeta_x = self.config.rho_x * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        zeta_v = self.config.rho_v * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        beta_x = self.config.rho_x * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        beta_v = self.config.rho_v * torch.randn(horizon + 1, dtype=self.config.dtype, device=self.config.device, generator=generator)
        return torch.stack([zeta_x, zeta_v, beta_x, beta_v], dim=-1)

    def _perturb_states(
        self,
        states: torch.Tensor,
        perturbation: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor:
        if lambda_value == 0.0 or (self.config.rho_x == 0.0 and self.config.rho_v == 0.0):
            return states
        zeta_x, zeta_v, beta_x, beta_v = perturbation.unbind(dim=-1)
        x = (1.0 + lambda_value * zeta_x) * states[..., 0] + lambda_value * beta_x
        v = (1.0 + lambda_value * zeta_v) * states[..., 1] + lambda_value * beta_v
        return torch.stack([x, v], dim=-1)

    def _generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator = torch.Generator(device=self.config.device)
        generator.manual_seed(int(seed))
        return generator
