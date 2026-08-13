from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import torch

from .registry import EnvironmentSpec


def initialize_control(spec: EnvironmentSpec, env: Any) -> torch.Tensor | torch.nn.Module:
    if spec.name == "twostate":
        return torch.nn.Parameter(torch.zeros(env.n_states, dtype=env.config.dtype, device=env.config.device))
    if spec.policy_cls is not None:
        if hasattr(env, "zero_policy"):
            policy = env.zero_policy()
        else:
            policy = spec.policy_cls(env.config)
        policy.train(True)
        for parameter in policy.parameters():
            parameter.requires_grad_(True)
        return policy
    if hasattr(env, "zero_policy"):
        return torch.nn.Parameter(env.zero_policy().detach().clone())
    raise ValueError(f"Environment {spec.name!r} does not define a default control.")


def control_payload(control: torch.Tensor | torch.nn.Module) -> Dict[str, Any]:
    if isinstance(control, torch.nn.Module):
        return {
            "kind": "module",
            "state_dict": {key: value.detach().cpu().clone() for key, value in control.state_dict().items()},
        }
    return {"kind": "tensor", "theta": control.detach().cpu().clone()}


def load_control(spec: EnvironmentSpec, env: Any, payload: Mapping[str, Any], trainable: bool = True) -> torch.Tensor | torch.nn.Module:
    kind = payload.get("kind")
    if kind == "module":
        if spec.policy_cls is None:
            raise ValueError(f"Checkpoint stores a module, but {spec.name!r} has no policy class.")
        policy = spec.policy_cls(env.config)
        state = {
            key: value.to(dtype=env.config.dtype, device=env.config.device)
            for key, value in payload["state_dict"].items()
        }
        policy.load_state_dict(state)
        policy.train(trainable)
        for parameter in policy.parameters():
            parameter.requires_grad_(trainable)
        return policy
    if kind == "tensor":
        theta = payload["theta"].to(dtype=env.config.dtype, device=env.config.device).detach().clone()
        return torch.nn.Parameter(theta) if trainable else theta
    raise ValueError(f"Unknown control payload kind {kind!r}.")


def control_parameters(control: torch.Tensor | torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    if isinstance(control, torch.nn.Module):
        return control.parameters()
    if isinstance(control, torch.nn.Parameter):
        return [control]
    return [torch.nn.Parameter(control)]


def control_vector(control: torch.Tensor | torch.nn.Module) -> torch.Tensor:
    if isinstance(control, torch.nn.Module):
        return torch.nn.utils.parameters_to_vector(control.parameters()).detach()
    return control.detach().reshape(-1)


def assign_gradient(control: torch.Tensor | torch.nn.Module, grad: torch.Tensor, objective: str) -> None:
    sign = -1.0 if objective == "maximize" else 1.0
    if isinstance(control, torch.nn.Module):
        flat = (sign * grad.detach().reshape(-1)).to(device=next(control.parameters()).device)
        offset = 0
        for parameter in control.parameters():
            count = parameter.numel()
            parameter.grad = flat[offset : offset + count].reshape_as(parameter).clone()
            offset += count
        if offset != flat.numel():
            raise ValueError(f"Gradient length {flat.numel()} does not match policy parameter length {offset}.")
        return
    control.grad = (sign * grad.detach()).reshape_as(control).clone()


__all__ = [
    "assign_gradient",
    "control_parameters",
    "control_payload",
    "control_vector",
    "initialize_control",
    "load_control",
]
