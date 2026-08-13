from __future__ import annotations

import math
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

import torch


class ContinuousTransportMFREINFORCE:
    """
    Transport-score MF-REINFORCE for continuous-state benchmarks.

    The implementation follows the single-batch forward estimator from the
    continuous-state algorithm: a nominal coordinate flow is estimated once,
    one auxiliary batch estimates the full coordinate-sensitivity flow, and one
    main batch reuses that sensitivity flow in the perturbed policy-gradient
    estimator.
    """

    def __init__(self, env: Any, algorithm_config: Optional[Mapping[str, Any]] = None):
        self.env = env
        self.config = env.config
        self.algorithm_config = dict(algorithm_config or {})
        self.env_kind = type(env).__name__

    def parameter_vector(self, control: torch.Tensor | torch.nn.Module) -> torch.Tensor:
        if isinstance(control, torch.nn.Module):
            return torch.nn.utils.parameters_to_vector(control.parameters()).detach()
        return control.detach().reshape(-1)

    def format_gradient(self, control: torch.Tensor | torch.nn.Module, grad_flat: torch.Tensor) -> torch.Tensor:
        if isinstance(control, torch.nn.Module):
            return grad_flat
        return grad_flat.reshape_as(control)

    def coordinate_dim(self) -> int:
        if self.env_kind in {"LinearQuadraticMFC", "MeanVariancePortfolioMFC"}:
            return 1
        if self.env_kind == "CuckerSmaleMFC":
            return 2
        if self.env_kind == "KuramotoMFC":
            return 2
        raise ValueError(f"{self.env_kind} does not expose continuous MF-REINFORCE coordinates.")

    def objective_kind(self) -> Literal["cost", "reward"]:
        configured = self.algorithm_config.get("objective")
        if configured in {"cost", "reward"}:
            return configured
        if self.env_kind == "MeanVariancePortfolioMFC":
            return "reward"
        return "cost"

    def default_horizon(self) -> int:
        return int(getattr(self.config, "T"))

    def score_chunk_size(self, param_dim: int, batch_size: int) -> int:
        configured = self.algorithm_config.get("score_chunk_size", getattr(self.config, "score_chunk_size", None))
        if configured is not None:
            return int(configured)
        if param_dim > 20_000:
            return min(batch_size, 16)
        return min(batch_size, 64)

    def perturbation_std(self) -> float:
        return float(self.algorithm_config.get("perturbation_std", 1.0))

    def keep_score_diagnostics(self, keep_score_diagnostics: Optional[bool]) -> bool:
        if keep_score_diagnostics is None:
            return bool(self.algorithm_config.get("keep_score_diagnostics", getattr(self.config, "keep_score_diagnostics", False)))
        return bool(keep_score_diagnostics)

    @torch.no_grad()
    def estimate_coordinate_flow(
        self,
        control: torch.Tensor | torch.nn.Module,
        *,
        horizon: Optional[int] = None,
        particles: Optional[int] = None,
        seed: Optional[int] = None,
        exploration: Optional[bool] = None,
    ) -> Dict[str, Any]:
        steps = self.default_horizon() if horizon is None else int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive.")

        if self.env_kind == "LinearQuadraticMFC":
            means = self.env.exact_moments(control)[0][: steps + 1].detach()
            return {"coordinates": means.unsqueeze(-1), "horizon": steps}

        if self.env_kind == "MeanVariancePortfolioMFC":
            means = self.env.exact_moments(control, lambda_=0.0)[0][: steps + 1].detach()
            return {"coordinates": means.unsqueeze(-1), "horizon": steps}

        n = int(particles or self.algorithm_config.get("population_particles", getattr(self.config, "N_pop", 32)))
        if self.env_kind == "CuckerSmaleMFC":
            rollout = self.env.sample_trajectories(
                control,
                n,
                seed=seed,
                lambda_=0.0,
                horizon=steps,
                exploration=bool(True if exploration is None else exploration),
            )
            law_flow = rollout["state_flow"][: steps + 1].detach()
            coordinates = law_flow.mean(dim=1)
            return {"coordinates": coordinates, "law_state_flow": law_flow, "horizon": steps}

        if self.env_kind == "KuramotoMFC":
            rollout = self.env.sample_trajectories(
                control,
                n,
                seed=seed,
                lambda_=0.0,
                horizon=steps,
                exploration=bool(True if exploration is None else exploration),
            )
            lifted = rollout["lifted_phase_flow"][: steps + 1].detach()
            coordinates = torch.stack([torch.cos(lifted).mean(dim=1), torch.sin(lifted).mean(dim=1)], dim=-1)
            return {"coordinates": coordinates, "horizon": steps}

        raise ValueError(f"{self.env_kind} does not support coordinate-flow estimation.")

    def estimate_sensitivity(
        self,
        control: torch.Tensor | torch.nn.Module,
        nominal: Mapping[str, Any] | torch.Tensor,
        lambda_: float,
        n_aux: int,
        *,
        seed: Optional[int] = None,
        baseline: Union[None, Literal["nominal"], torch.Tensor] = "nominal",
    ) -> torch.Tensor:
        if n_aux <= 0:
            raise ValueError("n_aux must be positive.")
        lambda_value = float(lambda_)
        if lambda_value <= 0.0:
            raise ValueError("lambda_ must be positive for transport-score sensitivity estimation.")

        nominal_data = self._as_nominal(nominal)
        coordinates = nominal_data["coordinates"]
        horizon = int(nominal_data.get("horizon", coordinates.shape[0] - 1))
        param_dim = self.parameter_vector(control).numel()
        coord_dim = self.coordinate_dim()
        generator = self._generator(seed)

        states, extra = self._sample_initial(n_aux, generator)
        cumulative_score = torch.zeros(n_aux, param_dim, dtype=self.config.dtype, device=self.config.device)
        sensitivity = torch.zeros(
            horizon + 1,
            coord_dim,
            param_dim,
            dtype=self.config.dtype,
            device=self.config.device,
        )

        for t in range(horizon + 1):
            psi = self._moment_statistics(states, extra)
            center = self._coordinate_baseline(coordinates[t], baseline)
            sensitivity[t] = ((psi - center).transpose(0, 1) @ cumulative_score) / float(n_aux)

            if t == horizon:
                break

            z, xi = self._sample_coordinate_noise(n_aux, coord_dim, generator)
            perturbed_coordinates = coordinates[t].unsqueeze(0) + lambda_value * z
            law = self._law_from_coordinates(nominal_data, t, perturbed_coordinates)
            actions, action_means = self._sample_actions(control, t, states, law, extra, generator)
            policy_scores = self._policy_scores(
                control,
                t,
                states,
                law,
                actions,
                action_means,
                extra,
                chunk_size=self.score_chunk_size(param_dim, n_aux),
            )
            population_scores = torch.einsum("kd,nk->nd", sensitivity[t], xi) / lambda_value
            cumulative_score = cumulative_score + policy_scores + population_scores
            states = self._transition(t, states, actions, law, extra, generator)

        return sensitivity.detach()

    def gradient_estimate(
        self,
        control: torch.Tensor | torch.nn.Module,
        nominal: Mapping[str, Any] | torch.Tensor,
        sensitivity: torch.Tensor,
        lambda_: float,
        B: int,
        *,
        seed: Optional[int] = None,
        baseline: Union[None, float, Literal["batch_mean", "time_batch_mean"]] = "batch_mean",
        keep_score_diagnostics: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if B <= 0:
            raise ValueError("B must be positive.")
        lambda_value = float(lambda_)
        if lambda_value <= 0.0:
            raise ValueError("lambda_ must be positive for transport-score gradient estimation.")

        nominal_data = self._as_nominal(nominal)
        coordinates = nominal_data["coordinates"]
        horizon = int(nominal_data.get("horizon", coordinates.shape[0] - 1))
        param_dim = self.parameter_vector(control).numel()
        coord_dim = self.coordinate_dim()
        generator = self._generator(seed)
        keep_scores = self.keep_score_diagnostics(keep_score_diagnostics)

        sensitivity = torch.as_tensor(sensitivity, dtype=self.config.dtype, device=self.config.device)
        expected_shape = (horizon + 1, coord_dim, param_dim)
        if tuple(sensitivity.shape) != expected_shape:
            raise ValueError(f"sensitivity must have shape {expected_shape}, got {tuple(sensitivity.shape)}.")

        states, extra = self._sample_initial(B, generator)
        stage_signals = torch.zeros(B, horizon, dtype=self.config.dtype, device=self.config.device)
        terminal_signal = torch.zeros(B, dtype=self.config.dtype, device=self.config.device)
        population_scores = torch.zeros(B, horizon + 1, param_dim, dtype=self.config.dtype, device=self.config.device)
        score_sum_samples = torch.zeros(B, param_dim, dtype=self.config.dtype, device=self.config.device) if keep_scores else None
        policy_score_sum = torch.zeros(param_dim, dtype=self.config.dtype, device=self.config.device)

        stored: list[Dict[str, Any]] = []
        for t in range(horizon):
            z, xi = self._sample_coordinate_noise(B, coord_dim, generator)
            perturbed_coordinates = coordinates[t].unsqueeze(0) + lambda_value * z
            law = self._law_from_coordinates(nominal_data, t, perturbed_coordinates)
            actions, action_means = self._sample_actions(control, t, states, law, extra, generator)
            population_scores[:, t] = torch.einsum("kd,bk->bd", sensitivity[t], xi) / lambda_value
            stage_signals[:, t] = self._running_signal(t, states, actions, law)
            stored.append(
                {
                    "t": t,
                    "states": states.detach(),
                    "law": self._detach_law(law),
                    "actions": actions.detach(),
                    "action_means": action_means.detach(),
                    "extra": self._slice_extra(extra, detach=True),
                }
            )
            states = self._transition(t, states, actions, law, extra, generator)

        z_terminal, xi_terminal = self._sample_coordinate_noise(B, coord_dim, generator)
        terminal_coordinates = coordinates[horizon].unsqueeze(0) + lambda_value * z_terminal
        terminal_law = self._law_from_coordinates(nominal_data, horizon, terminal_coordinates)
        population_scores[:, horizon] = torch.einsum("kd,bk->bd", sensitivity[horizon], xi_terminal) / lambda_value
        terminal_signal = self._terminal_signal(states, terminal_law)

        returns_to_go = torch.zeros(B, horizon + 1, dtype=self.config.dtype, device=self.config.device)
        returns_to_go[:, horizon] = terminal_signal
        for t in range(horizon - 1, -1, -1):
            returns_to_go[:, t] = stage_signals[:, t] + returns_to_go[:, t + 1]
        centered_returns = self._center_returns(returns_to_go, baseline)

        score_chunk_size = self.score_chunk_size(param_dim, B)
        for record in stored:
            t = int(record["t"])
            weights = centered_returns[:, t]
            policy_score_sum = policy_score_sum + self._weighted_policy_score_sums(
                control,
                t,
                record["states"],
                record["law"],
                record["actions"],
                weights,
                record["extra"],
                chunk_size=score_chunk_size,
            ).reshape(param_dim)
            if keep_scores and score_sum_samples is not None:
                per_sample = self._policy_scores(
                    control,
                    t,
                    record["states"],
                    record["law"],
                    record["actions"],
                    record["action_means"],
                    record["extra"],
                    chunk_size=score_chunk_size,
                )
                score_sum_samples = score_sum_samples + per_sample + population_scores[:, t]

        population_score_sum = torch.einsum("bt,btd->d", centered_returns, population_scores)
        if keep_scores and score_sum_samples is not None:
            score_sum_samples = score_sum_samples + population_scores[:, horizon]

        grad_flat = (policy_score_sum + population_score_sum) / float(B)
        grad_hat = self.format_gradient(control, grad_flat)
        returns = returns_to_go[:, 0]
        diag: Dict[str, torch.Tensor] = {
            "returns": returns.detach(),
            "mean_return": returns.mean().detach(),
            "std_return": returns.std(unbiased=False).detach(),
            "grad_norm": torch.linalg.norm(grad_flat.detach()),
            "lambda": torch.tensor(lambda_value, dtype=self.config.dtype, device=self.config.device),
        }
        if keep_scores and score_sum_samples is not None:
            diag["scores"] = score_sum_samples.detach()
        return grad_hat, diag

    def complete_gradient_estimate(
        self,
        control: torch.Tensor | torch.nn.Module,
        lambda_: float,
        B: int,
        n_aux: int,
        *,
        eta: Optional[float] = None,
        horizon: Optional[int] = None,
        nominal: Optional[Mapping[str, Any] | torch.Tensor] = None,
        population_particles: Optional[int] = None,
        seed: Optional[int] = None,
        baseline: Union[None, float, Literal["batch_mean", "time_batch_mean"]] = "batch_mean",
        sensitivity_baseline: Union[None, Literal["nominal"], torch.Tensor] = "nominal",
        keep_score_diagnostics: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lambda_value = float(lambda_)
        eta_value = lambda_value if eta is None else float(eta)
        if nominal is None:
            nominal = self.estimate_coordinate_flow(
                control,
                horizon=horizon,
                particles=population_particles,
                seed=seed,
                exploration=self.algorithm_config.get("coordinate_exploration"),
            )
        sensitivity = self.estimate_sensitivity(
            control,
            nominal,
            eta_value,
            n_aux,
            seed=None if seed is None else int(seed) + 1_000_003,
            baseline=sensitivity_baseline,
        )
        grad_hat, diag = self.gradient_estimate(
            control,
            nominal,
            sensitivity,
            lambda_value,
            B,
            seed=None if seed is None else int(seed) + 2_000_003,
            baseline=baseline,
            keep_score_diagnostics=keep_score_diagnostics,
        )
        diag = dict(diag)
        diag["sensitivity"] = sensitivity
        diag["eta"] = torch.tensor(eta_value, dtype=self.config.dtype, device=self.config.device)
        diag["main_trajectories"] = torch.tensor(int(B), device=self.config.device)
        diag["auxiliary_trajectories"] = torch.tensor(int(n_aux), device=self.config.device)
        if isinstance(nominal, Mapping):
            diag["coordinate_flow"] = nominal["coordinates"].detach()
        return grad_hat, diag

    def _as_nominal(self, nominal: Mapping[str, Any] | torch.Tensor) -> Dict[str, Any]:
        if isinstance(nominal, Mapping):
            out = dict(nominal)
            out["coordinates"] = torch.as_tensor(out["coordinates"], dtype=self.config.dtype, device=self.config.device)
            if "law_state_flow" in out:
                out["law_state_flow"] = torch.as_tensor(out["law_state_flow"], dtype=self.config.dtype, device=self.config.device)
            out.setdefault("horizon", out["coordinates"].shape[0] - 1)
            return out
        coordinates = torch.as_tensor(nominal, dtype=self.config.dtype, device=self.config.device)
        if coordinates.ndim == 1:
            coordinates = coordinates.unsqueeze(-1)
        return {"coordinates": coordinates, "horizon": coordinates.shape[0] - 1}

    def _coordinate_baseline(
        self,
        coordinate: torch.Tensor,
        baseline: Union[None, Literal["nominal"], torch.Tensor],
    ) -> torch.Tensor:
        if baseline == "nominal":
            return coordinate.unsqueeze(0)
        if baseline is None:
            return torch.zeros((1, coordinate.numel()), dtype=self.config.dtype, device=self.config.device)
        value = torch.as_tensor(baseline, dtype=self.config.dtype, device=self.config.device)
        return value.reshape(1, -1)

    def _sample_coordinate_noise(
        self,
        n: int,
        coord_dim: int,
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.perturbation_std()
        if std <= 0.0:
            raise ValueError("perturbation_std must be positive.")
        if bool(self.algorithm_config.get("antithetic", False)) and n > 1:
            half = n // 2
            base = std * torch.randn(half, coord_dim, dtype=self.config.dtype, device=self.config.device, generator=generator)
            pieces = [base, -base]
            if n % 2:
                pieces.append(std * torch.randn(1, coord_dim, dtype=self.config.dtype, device=self.config.device, generator=generator))
            z = torch.cat(pieces, dim=0)
        else:
            z = std * torch.randn(n, coord_dim, dtype=self.config.dtype, device=self.config.device, generator=generator)
        xi = z / (std**2)
        return z, xi

    def _law_from_coordinates(self, nominal: Mapping[str, Any], t: int, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = torch.as_tensor(coordinates, dtype=self.config.dtype, device=self.config.device)
        if self.env_kind in {"LinearQuadraticMFC", "MeanVariancePortfolioMFC", "KuramotoMFC"}:
            return coordinates
        if self.env_kind == "CuckerSmaleMFC":
            law_flow = nominal.get("law_state_flow")
            if law_flow is None:
                raise ValueError("Cucker-Smale continuous MF-REINFORCE requires nominal['law_state_flow'].")
            base_law = law_flow[t]
            base_coordinate = nominal["coordinates"][t]
            delta = coordinates - base_coordinate.unsqueeze(0)
            return base_law.unsqueeze(0) + delta.unsqueeze(1)
        raise ValueError(f"Unsupported law coordinate chart for {self.env_kind}.")

    @torch.no_grad()
    def _sample_initial(
        self,
        n: int,
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.env_kind == "LinearQuadraticMFC":
            x = self.config.x0_mean + self.config.x0_std * torch.randn(
                n,
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            return x, {}
        if self.env_kind == "MeanVariancePortfolioMFC":
            x = self.config.x0_mean + self.config.x0_std * torch.randn(
                n,
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            return x, {}
        if self.env_kind == "CuckerSmaleMFC":
            return self.env.sample_initial_states(n, generator=generator), {}
        if self.env_kind == "KuramotoMFC":
            phases = self.env.sample_initial_phases(n, generator=generator)
            frequencies = self.env.sample_frequencies(n, generator=generator)
            return phases, {"frequencies": frequencies}
        raise ValueError(f"Unsupported initial sampler for {self.env_kind}.")

    def _moment_statistics(self, states: torch.Tensor, extra: Mapping[str, torch.Tensor]) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=self.config.dtype, device=self.config.device)
        if self.env_kind in {"LinearQuadraticMFC", "MeanVariancePortfolioMFC"}:
            return states.reshape(-1, 1)
        if self.env_kind == "CuckerSmaleMFC":
            return states.reshape(states.shape[0], 2)
        if self.env_kind == "KuramotoMFC":
            return torch.stack([torch.cos(states), torch.sin(states)], dim=-1)
        raise ValueError(f"Unsupported moment statistic for {self.env_kind}.")

    @torch.no_grad()
    def _sample_actions(
        self,
        control: torch.Tensor | torch.nn.Module,
        t: int,
        states: torch.Tensor,
        law: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.env_kind == "LinearQuadraticMFC":
            law_mean = law.reshape(-1)
            action_mean = self.env.policy_mean(control, t, states, law_mean)
            residual = self.config.policy_std * torch.randn_like(action_mean, generator=generator)
            return action_mean + residual, action_mean
        if self.env_kind == "MeanVariancePortfolioMFC":
            law_mean = law.reshape(-1)
            action_mean = self.env.policy_mean(control, t, states, law_mean)
            residual = self.env.tau[t] * torch.randn_like(action_mean, generator=generator)
            return action_mean + residual, action_mean
        if self.env_kind == "CuckerSmaleMFC":
            action_mean = self._cucker_policy_mean(control, t, states, law)
            residual = self.config.tau * torch.randn_like(action_mean, generator=generator)
            return action_mean + residual, action_mean
        if self.env_kind == "KuramotoMFC":
            action_mean = self._kuramoto_policy_mean(control, t, states, law, extra.get("frequencies"))
            residual = self.config.tau * torch.randn_like(action_mean, generator=generator)
            return action_mean + residual, action_mean
        raise ValueError(f"Unsupported action sampler for {self.env_kind}.")

    def _policy_scores(
        self,
        control: torch.Tensor | torch.nn.Module,
        t: int,
        states: torch.Tensor,
        law: torch.Tensor,
        actions: torch.Tensor,
        action_means: Optional[torch.Tensor],
        extra: Mapping[str, torch.Tensor],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        if not isinstance(control, torch.nn.Module):
            if self.env_kind in {"LinearQuadraticMFC", "MeanVariancePortfolioMFC"}:
                scores = self.env.policy_score_batch(
                    control,
                    t,
                    states,
                    law.reshape(-1),
                    actions,
                    action_means=action_means,
                )
                return scores.reshape(states.shape[0], -1).detach()
            raise ValueError(f"Tensor controls are not supported for {self.env_kind}.")

        params = tuple(control.parameters())
        batch = states.shape[0]
        chunks = []
        for start in range(0, batch, chunk_size):
            end = min(start + chunk_size, batch)
            logp = self._module_log_probs(
                control,
                t,
                self._slice_batch(states, start, end),
                self._slice_batch(law, start, end),
                self._slice_batch(actions, start, end),
                self._slice_extra(extra, start, end),
            )
            grad_outputs = torch.eye(end - start, dtype=self.config.dtype, device=self.config.device)
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                allow_unused=True,
            )
            flat = [
                torch.zeros((end - start, parameter.numel()), dtype=self.config.dtype, device=self.config.device)
                if grad is None
                else grad.reshape(end - start, -1)
                for grad, parameter in zip(grads, params)
            ]
            chunks.append(torch.cat(flat, dim=1).detach())
        return torch.cat(chunks, dim=0)

    def _weighted_policy_score_sums(
        self,
        control: torch.Tensor | torch.nn.Module,
        t: int,
        states: torch.Tensor,
        law: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        weights = weights.to(dtype=self.config.dtype, device=self.config.device)
        if not isinstance(control, torch.nn.Module):
            scores = self._policy_scores(control, t, states, law, actions, None, extra, chunk_size=chunk_size)
            weights_2d = weights.reshape(-1, scores.shape[0])
            result = weights_2d @ scores
            return result.squeeze(0) if weights.ndim == 1 else result

        params = tuple(control.parameters())
        states_flat = states.reshape(states.shape[0], *states.shape[1:])
        weights_2d = weights.reshape(-1, states_flat.shape[0])
        param_dim = sum(parameter.numel() for parameter in params)
        result = torch.zeros(weights_2d.shape[0], param_dim, dtype=self.config.dtype, device=self.config.device)

        for start in range(0, states_flat.shape[0], chunk_size):
            end = min(start + chunk_size, states_flat.shape[0])
            chunk_weights = weights_2d[:, start:end]
            active = chunk_weights.abs().sum(dim=1) > 0
            if not bool(active.any()):
                continue
            logp = self._module_log_probs(
                control,
                t,
                self._slice_batch(states, start, end),
                self._slice_batch(law, start, end),
                self._slice_batch(actions, start, end),
                self._slice_extra(extra, start, end),
            )
            grads = torch.autograd.grad(
                logp,
                params,
                grad_outputs=chunk_weights[active],
                is_grads_batched=True,
                allow_unused=True,
            )
            flat = [
                torch.zeros((int(active.sum().item()), parameter.numel()), dtype=self.config.dtype, device=self.config.device)
                if grad is None
                else grad.reshape(int(active.sum().item()), -1)
                for grad, parameter in zip(grads, params)
            ]
            result[active] += torch.cat(flat, dim=1).detach()
        return result.squeeze(0) if weights.ndim == 1 else result

    def _module_log_probs(
        self,
        control: torch.nn.Module,
        t: int,
        states: torch.Tensor,
        law: torch.Tensor,
        actions: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.env_kind == "CuckerSmaleMFC":
            action_mean = self._cucker_policy_mean(control, t, states, law)
            residual = actions - action_mean
            tau_sq = self.config.tau**2
            return -0.5 * (residual.square() / tau_sq + math.log(2.0 * math.pi * tau_sq))
        if self.env_kind == "KuramotoMFC":
            action_mean = self._kuramoto_policy_mean(control, t, states, law, extra.get("frequencies"))
            residual = actions - action_mean
            tau_sq = self.config.tau**2
            return -0.5 * (residual.square() / tau_sq + math.log(2.0 * math.pi * tau_sq))
        raise ValueError(f"Module policy scores are not supported for {self.env_kind}.")

    @torch.no_grad()
    def _transition(
        self,
        t: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        law: torch.Tensor,
        extra: Mapping[str, torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.env_kind == "LinearQuadraticMFC":
            law_mean = law.reshape(-1)
            noise = self.config.noise_std * torch.randn_like(states, generator=generator)
            return self.config.a * states + self.config.b * actions + self.config.c * law_mean + noise
        if self.env_kind == "MeanVariancePortfolioMFC":
            excess_return = self._sample_portfolio_excess_returns(t, states.shape[0], generator)
            return self.env.s[t] * states + excess_return * actions
        if self.env_kind == "CuckerSmaleMFC":
            alignment = self._cucker_alignment_field(states, law)
            noise = self.config.sigma * (self.config.dt**0.5) * torch.randn(
                states.shape[:-1],
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            x_next = states[..., 0] + self.config.dt * states[..., 1]
            v_next = states[..., 1] + self.config.dt * (alignment + actions) + noise
            return torch.stack([x_next, v_next], dim=-1)
        if self.env_kind == "KuramotoMFC":
            frequencies = extra.get("frequencies")
            drift = torch.zeros_like(states) if frequencies is None else frequencies
            interaction = self._kuramoto_interaction_field(states, law)
            noise_scale = torch.sqrt(
                torch.as_tensor(2.0 * self.config.D * self.config.dt, dtype=self.config.dtype, device=self.config.device)
            )
            noise = noise_scale * torch.randn(states.shape, dtype=self.config.dtype, device=self.config.device, generator=generator)
            return states + self.config.dt * (drift + self.config.K * interaction + actions) + noise
        raise ValueError(f"Unsupported transition for {self.env_kind}.")

    def _running_signal(self, t: int, states: torch.Tensor, actions: torch.Tensor, law: torch.Tensor) -> torch.Tensor:
        if self.env_kind == "LinearQuadraticMFC":
            return self.env.running_cost_batch(states, actions, law.reshape(-1))
        if self.env_kind == "MeanVariancePortfolioMFC":
            return self.env.running_reward_batch(states, actions, law.reshape(-1))
        if self.env_kind == "CuckerSmaleMFC":
            v_mean = law[..., 1].mean(dim=-1)
            v_error = states[..., 1] - v_mean
            return self.config.dt * (v_error.square() + self.config.gamma * actions.square())
        if self.env_kind == "KuramotoMFC":
            c = law[..., 0]
            s = law[..., 1]
            local_disagreement = 1.0 - torch.cos(states) * c - torch.sin(states) * s
            target_lock = 1.0 - torch.cos(states - self.config.theta_star)
            return self.config.dt * (
                self.config.kappa_sync * local_disagreement
                + self.config.kappa_lock * target_lock
                + self.config.gamma * actions.square()
            )
        raise ValueError(f"Unsupported running signal for {self.env_kind}.")

    def _terminal_signal(self, states: torch.Tensor, law: torch.Tensor) -> torch.Tensor:
        if self.env_kind == "LinearQuadraticMFC":
            return self.env.terminal_cost_batch(states, law.reshape(-1))
        if self.env_kind == "MeanVariancePortfolioMFC":
            return self.env.terminal_reward_batch(states, law.reshape(-1))
        if self.env_kind == "CuckerSmaleMFC":
            v_mean = law[..., 1].mean(dim=-1)
            return self.config.kappa_T * (states[..., 1] - v_mean).square()
        if self.env_kind == "KuramotoMFC":
            c = law[..., 0]
            s = law[..., 1]
            local_disagreement = 1.0 - torch.cos(states) * c - torch.sin(states) * s
            target_lock = 1.0 - torch.cos(states - self.config.theta_star)
            return self.config.kappa_sync_T * local_disagreement + self.config.kappa_lock_T * target_lock
        raise ValueError(f"Unsupported terminal signal for {self.env_kind}.")

    def _center_returns(
        self,
        returns_to_go: torch.Tensor,
        baseline: Union[None, float, Literal["batch_mean", "time_batch_mean"]],
    ) -> torch.Tensor:
        if baseline == "batch_mean":
            return returns_to_go - returns_to_go[:, 0].mean()
        if baseline == "time_batch_mean":
            return returns_to_go - returns_to_go.mean(dim=0, keepdim=True)
        if baseline is None:
            return returns_to_go
        return returns_to_go - torch.tensor(float(baseline), dtype=self.config.dtype, device=self.config.device)

    def _cucker_alignment_field(self, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        x = states[..., 0]
        v = states[..., 1]
        law_x = law_states[..., 0]
        law_v = law_states[..., 1]
        distances = torch.abs(x.unsqueeze(-1) - law_x)
        weights = self.env.communication_kernel(distances)
        return (weights * (law_v - v.unsqueeze(-1))).mean(dim=-1)

    def _cucker_policy_mean(self, control: torch.nn.Module, t: int, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        features = self._cucker_policy_features(t, states, law_states)
        return control(features)

    def _cucker_policy_features(self, t: int, states: torch.Tensor, law_states: torch.Tensor) -> torch.Tensor:
        x = states[..., 0]
        v = states[..., 1]
        x_mean = law_states[..., 0].mean(dim=-1)
        v_mean = law_states[..., 1].mean(dim=-1)
        v_variance = (law_states[..., 1] - v_mean.unsqueeze(-1)).square().mean(dim=-1)
        alignment = self._cucker_alignment_field(states, law_states)
        time = torch.full_like(x, float(t) / max(1, self.config.T))
        return torch.stack(
            [
                time,
                x,
                v,
                x_mean,
                v_mean,
                alignment,
                torch.sqrt(v_variance + self.config.eps_num),
            ],
            dim=-1,
        )

    def _kuramoto_interaction_field(self, phases: torch.Tensor, law_coordinates: torch.Tensor) -> torch.Tensor:
        c = law_coordinates[..., 0]
        s = law_coordinates[..., 1]
        return s * torch.cos(phases) - c * torch.sin(phases)

    def _kuramoto_policy_mean(
        self,
        control: torch.nn.Module,
        t: int,
        phases: torch.Tensor,
        law_coordinates: torch.Tensor,
        frequencies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        features = self._kuramoto_policy_features(t, phases, law_coordinates, frequencies)
        return control(features)

    def _kuramoto_policy_features(
        self,
        t: int,
        phases: torch.Tensor,
        law_coordinates: torch.Tensor,
        frequencies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c = law_coordinates[..., 0]
        s = law_coordinates[..., 1]
        r = torch.sqrt(c.square() + s.square()).clamp_min(0.0)
        interaction = self._kuramoto_interaction_field(phases, law_coordinates)
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
        if self.config.uses_intrinsic_frequencies:
            if frequencies is None:
                features.append(torch.zeros_like(phases))
            else:
                features.append(torch.broadcast_to(frequencies, phases.shape))
        return torch.stack(features, dim=-1)

    def _sample_portfolio_excess_returns(
        self,
        t: int,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.config.return_distribution == "normal":
            standardized = torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
        else:
            standardized = self._sample_standardized_student_t(n, generator)
        return self.env.r_bar[t] + self.env.sigma_R[t] * standardized

    def _sample_standardized_student_t(self, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        df = float(self.config.student_t_df)
        df_int = int(round(df))
        if abs(df - df_int) > 1e-12:
            distribution = torch.distributions.StudentT(torch.as_tensor(df, dtype=self.config.dtype, device=self.config.device))
            sample = distribution.sample((n,))
        else:
            numerator = torch.randn(n, dtype=self.config.dtype, device=self.config.device, generator=generator)
            denominator_normals = torch.randn(
                (n, df_int),
                dtype=self.config.dtype,
                device=self.config.device,
                generator=generator,
            )
            chi_square = denominator_normals.square().sum(dim=-1)
            sample = numerator / torch.sqrt(chi_square / df)
        return sample * torch.sqrt(torch.as_tensor((df - 2.0) / df, dtype=self.config.dtype, device=self.config.device))

    def _slice_batch(self, value: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return value[start:end]

    def _slice_extra(
        self,
        extra: Mapping[str, torch.Tensor],
        start: Optional[int] = None,
        end: Optional[int] = None,
        *,
        detach: bool = False,
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for key, value in extra.items():
            sliced = value if start is None else value[start:end]
            result[key] = sliced.detach() if detach else sliced
        return result

    def _detach_law(self, law: torch.Tensor) -> torch.Tensor:
        return law.detach()

    def _generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator = torch.Generator(device=self.config.device)
        generator.manual_seed(int(seed))
        return generator


__all__ = ["ContinuousTransportMFREINFORCE"]
