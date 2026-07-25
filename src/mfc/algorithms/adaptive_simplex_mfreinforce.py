from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Literal, Optional, Tuple, Union

import torch

from .simplex_mfreinforce import SimplexPerturbedMFREINFORCE


@dataclass
class AdaptiveSimplexControllerConfig:
    initial_lambda: float = 0.4
    lambda_min: float = 0.01
    lambda_max: float = 0.8
    lambda_floor: float = 1e-8
    checkpoint_interval: int = 100
    diagnostic_replications: int = 4
    contraction: float = 0.5
    target_bias_variance_ratio: float = 1.0
    controller_lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    delta: float = 1e-12
    adam_eps: float = 1e-8
    bias_order: float = 1.0
    min_direction_cosine: float = 0.0
    direction_norm_threshold: float = 1e-12
    direction_pressure: float = 1.0
    eta_power: float = 1.0
    envelope_lambda0: float = 0.4
    envelope_m0: float = 1000.0
    envelope_zeta: float = 0.25
    main_sample_growth_power: float = 0.25
    aux_sample_growth_power: float = 0.5
    sample_growth_interval: float = 1000.0

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_min < self.lambda_max < 1.0):
            raise ValueError("Require 0 < lambda_min < lambda_max < 1.")
        if self.lambda_floor <= 0.0:
            raise ValueError("lambda_floor must be positive.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if self.diagnostic_replications <= 0:
            raise ValueError("diagnostic_replications must be positive.")
        if not (0.0 < self.contraction < 1.0):
            raise ValueError("contraction must lie in (0, 1).")
        if self.target_bias_variance_ratio <= 0.0:
            raise ValueError("target_bias_variance_ratio must be positive.")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must lie in [0, 1).")
        if self.bias_order <= 0.0:
            raise ValueError("bias_order must be positive.")
        if self.eta_power <= 0.0:
            raise ValueError("eta_power must be positive.")
        if self.envelope_lambda0 <= 0.0 or self.envelope_m0 <= 0.0 or self.envelope_zeta <= 0.0:
            raise ValueError("envelope_lambda0, envelope_m0, and envelope_zeta must be positive.")
        if self.sample_growth_interval <= 0.0:
            raise ValueError("sample_growth_interval must be positive.")


class FiniteBudgetAdaptiveSimplexMFREINFORCE:
    def __init__(self, env, controller_config: Optional[AdaptiveSimplexControllerConfig] = None):
        self.env = env
        self.config = env.config
        self.controller_config = controller_config or AdaptiveSimplexControllerConfig()
        self.simplex = SimplexPerturbedMFREINFORCE(env)
        self._moment_1 = 0.0
        self._moment_2 = 0.0
        self._checkpoint_count = 0
        self._rho = self._rho_from_lambda(self.controller_config.initial_lambda)
        self.lambda_ctrl = self._lambda_from_rho(self._rho)

    def parameter_vector(self, control) -> torch.Tensor:
        return self.simplex.parameter_vector(control)

    def format_gradient(self, control, grad_flat: torch.Tensor) -> torch.Tensor:
        return self.simplex.format_gradient(control, grad_flat)

    @torch.no_grad()
    def estimate_population_flow(
        self,
        control,
        mu0: torch.Tensor,
        n_particles: int,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        return self.simplex.estimate_population_flow(control, mu0, n_particles, horizon=horizon)

    def _rho_from_lambda(self, value: float) -> float:
        cfg = self.controller_config
        clipped = min(max(float(value), cfg.lambda_min), cfg.lambda_max)
        fraction = (clipped - cfg.lambda_min) / (cfg.lambda_max - cfg.lambda_min)
        fraction = min(max(fraction, 1e-12), 1.0 - 1e-12)
        return math.log(fraction / (1.0 - fraction))

    def _lambda_from_rho(self, rho: float) -> float:
        cfg = self.controller_config
        sigmoid = 1.0 / (1.0 + math.exp(-float(rho)))
        return cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * sigmoid

    def current_lambda(self, iteration: Optional[int] = None) -> float:
        return float(self.lambda_ctrl)

    def eta_for_lambda(self, lambda_value: float, iteration: Optional[int] = None) -> float:
        value = float(lambda_value) ** self.controller_config.eta_power
        return min(max(value, self.controller_config.lambda_floor), 1.0 - self.controller_config.lambda_floor)

    def sample_sizes(self, iteration: int, B: int, n_aux: int) -> Tuple[int, int]:
        return int(B), int(n_aux)

    def is_checkpoint(self, iteration: int) -> bool:
        return int(iteration) % self.controller_config.checkpoint_interval == 0

    def _torch_rng_snapshot(self) -> Dict[str, object]:
        snapshot: Dict[str, object] = {"cpu": torch.random.get_rng_state()}
        if self.config.device.type == "cuda":
            snapshot["cuda"] = torch.cuda.get_rng_state_all()
        return snapshot

    def _restore_torch_rng(self, snapshot: Dict[str, object]) -> None:
        torch.random.set_rng_state(snapshot["cpu"])
        if self.config.device.type == "cuda" and "cuda" in snapshot:
            torch.cuda.set_rng_state_all(snapshot["cuda"])

    def _flat_grad(self, grad_hat: torch.Tensor) -> torch.Tensor:
        return grad_hat.detach().reshape(-1)

    def _single_simplex_gradient(
        self,
        control,
        mu_flow: torch.Tensor,
        lambda_value: float,
        eta_value: float,
        B: int,
        n_aux: int,
        baseline: Union[None, float, Literal["batch_mean"]],
    ) -> torch.Tensor:
        grad_hat, _ = self.simplex.complete_gradient_estimate(
            control,
            mu_flow,
            lambda_value,
            B,
            n_aux,
            eta=eta_value,
            baseline=baseline,
        )
        return self._flat_grad(grad_hat)

    def checkpoint_diagnostic(
        self,
        control,
        mu_flow: torch.Tensor,
        lambda_value: float,
        B: int,
        n_aux: int,
        baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
    ) -> Dict[str, object]:
        cfg = self.controller_config
        plus_lambda = max(float(lambda_value), cfg.lambda_floor)
        minus_lambda = max(cfg.contraction * plus_lambda, cfg.lambda_floor)
        plus_eta = self.eta_for_lambda(plus_lambda)
        minus_eta = self.eta_for_lambda(minus_lambda)

        plus_samples = []
        minus_samples = []
        for _ in range(cfg.diagnostic_replications):
            snapshot = self._torch_rng_snapshot()
            plus_samples.append(
                self._single_simplex_gradient(control, mu_flow, plus_lambda, plus_eta, B, n_aux, baseline)
            )
            self._restore_torch_rng(snapshot)
            minus_samples.append(
                self._single_simplex_gradient(control, mu_flow, minus_lambda, minus_eta, B, n_aux, baseline)
            )

        plus = torch.stack(plus_samples)
        minus = torch.stack(minus_samples)
        delta_samples = plus - minus
        mean_plus = plus.mean(dim=0)
        mean_minus = minus.mean(dim=0)
        mean_delta = delta_samples.mean(dim=0)
        unbiased = cfg.diagnostic_replications > 1
        delta_cov_trace = float(delta_samples.var(dim=0, unbiased=unbiased).sum().item()) if unbiased else 0.0
        plus_cov_trace = float(plus.var(dim=0, unbiased=unbiased).sum().item()) if unbiased else 0.0

        discrepancy_sq = max(float(mean_delta.square().sum().item()) - delta_cov_trace / cfg.diagnostic_replications, 0.0)
        bias_denominator = (1.0 - cfg.contraction ** cfg.bias_order) ** 2
        bias_proxy_sq = discrepancy_sq / bias_denominator
        variance_proxy = plus_cov_trace
        z = math.log((bias_proxy_sq + cfg.delta) / (cfg.target_bias_variance_ratio * variance_proxy + cfg.delta))

        plus_norm = float(torch.linalg.norm(mean_plus).item())
        minus_norm = float(torch.linalg.norm(mean_minus).item())
        directional_cosine = float("nan")
        direction_triggered = False
        if plus_norm > cfg.direction_norm_threshold and minus_norm > cfg.direction_norm_threshold:
            directional_cosine = float((mean_plus @ mean_minus / (plus_norm * minus_norm)).item())
            if directional_cosine < cfg.min_direction_cosine:
                z = max(z, cfg.direction_pressure)
                direction_triggered = True

        self._update_controller(z)
        return {
            "lambda_plus": plus_lambda,
            "lambda_minus": minus_lambda,
            "eta_plus": plus_eta,
            "eta_minus": minus_eta,
            "bias_proxy_sq": bias_proxy_sq,
            "variance_proxy": variance_proxy,
            "delta_covariance_trace": delta_cov_trace,
            "mean_delta_norm": float(torch.linalg.norm(mean_delta).item()),
            "signed_pressure": z,
            "directional_cosine": directional_cosine,
            "direction_triggered": direction_triggered,
            "lambda_ctrl_after": self.lambda_ctrl,
            "rho_after": self._rho,
        }

    def _update_controller(self, signed_pressure: float) -> None:
        cfg = self.controller_config
        self._moment_1 = cfg.beta1 * self._moment_1 + (1.0 - cfg.beta1) * signed_pressure
        self._moment_2 = cfg.beta2 * self._moment_2 + (1.0 - cfg.beta2) * signed_pressure * signed_pressure
        step_index = self._checkpoint_count + 1
        moment_1_hat = self._moment_1 / (1.0 - cfg.beta1 ** step_index)
        moment_2_hat = self._moment_2 / (1.0 - cfg.beta2 ** step_index)
        self._rho = self._rho - cfg.controller_lr * moment_1_hat / (math.sqrt(moment_2_hat) + cfg.adam_eps)
        self.lambda_ctrl = self._lambda_from_rho(self._rho)
        self._checkpoint_count = step_index

    def gradient_estimate(
        self,
        control,
        mu_flow: torch.Tensor,
        iteration: int,
        B: int,
        n_aux: int,
        baseline: Union[None, float, Literal["batch_mean"]] = "batch_mean",
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        lambda_value = self.current_lambda(iteration)
        B_actual, n_actual = self.sample_sizes(iteration, B, n_aux)
        eta_value = self.eta_for_lambda(lambda_value, iteration)
        grad_hat, diag = self.simplex.complete_gradient_estimate(
            control,
            mu_flow,
            lambda_value,
            B_actual,
            n_actual,
            eta=eta_value,
            baseline=baseline,
        )

        horizon = mu_flow.shape[0] - 1
        base_transitions = horizon * (B_actual + n_actual)
        controller_diag = None
        diagnostic_transitions = 0
        if self.is_checkpoint(iteration):
            controller_diag = self.checkpoint_diagnostic(
                control,
                mu_flow,
                lambda_value,
                B_actual,
                n_actual,
                baseline=baseline,
            )
            diagnostic_transitions = 2 * self.controller_config.diagnostic_replications * base_transitions

        diag = dict(diag)
        diag.update(
            {
                "lambda": torch.tensor(lambda_value, dtype=self.config.dtype, device=self.config.device),
                "eta": torch.tensor(eta_value, dtype=self.config.dtype, device=self.config.device),
                "lambda_ctrl": torch.tensor(self.lambda_ctrl, dtype=self.config.dtype, device=self.config.device),
                "checkpoint": self.is_checkpoint(iteration),
                "controller": controller_diag,
                "main_trajectories": torch.tensor(B_actual, device=self.config.device),
                "auxiliary_trajectories": torch.tensor(n_actual, device=self.config.device),
                "simulator_transitions": torch.tensor(base_transitions + diagnostic_transitions, device=self.config.device),
            }
        )
        return grad_hat, diag


class ConsistentAdaptiveSimplexMFREINFORCE(FiniteBudgetAdaptiveSimplexMFREINFORCE):
    def current_lambda(self, iteration: Optional[int] = None) -> float:
        step = 0 if iteration is None else max(int(iteration), 0)
        envelope = self.controller_config.envelope_lambda0 / (
            1.0 + step / self.controller_config.envelope_m0
        ) ** self.controller_config.envelope_zeta
        return max(min(float(self.lambda_ctrl), float(envelope)), self.controller_config.lambda_floor)

    def sample_sizes(self, iteration: int, B: int, n_aux: int) -> Tuple[int, int]:
        cfg = self.controller_config
        step = max(int(iteration), 0)
        growth_base = 1.0 + step / cfg.sample_growth_interval
        B_actual = math.ceil(int(B) * growth_base ** cfg.main_sample_growth_power)
        n_actual = math.ceil(int(n_aux) * growth_base ** cfg.aux_sample_growth_power)
        return max(1, B_actual), max(1, n_actual)

    def eta_for_lambda(self, lambda_value: float, iteration: Optional[int] = None) -> float:
        value = float(lambda_value) ** self.controller_config.eta_power
        return min(max(value, self.controller_config.lambda_floor), 1.0 - self.controller_config.lambda_floor)
