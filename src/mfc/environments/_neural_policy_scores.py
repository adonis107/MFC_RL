from __future__ import annotations

import torch


def weighted_mlp_policy_score_sums(
    policy: torch.nn.Module,
    t: int,
    mu: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
    n_states: int,
    n_actions: int,
    time_scale: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    states_flat = states.reshape(-1)
    actions_flat = actions.reshape(-1)
    if mu.ndim == 1:
        mu_flat = mu.unsqueeze(0).expand(states_flat.numel(), n_states)
    else:
        mu_flat = mu.reshape(states_flat.numel(), n_states)

    weights_2d = weights.to(dtype=mu.dtype, device=mu.device).reshape(-1, states_flat.numel())
    group_shape = weights.shape[:-1] if weights.shape != states.shape else ()
    if chunk_size is None:
        chunk_size = states_flat.numel()

    linear1 = policy.net[0]
    linear2 = policy.net[2]
    linear3 = policy.net[4]
    params = tuple(policy.parameters())
    param_dim = sum(parameter.numel() for parameter in params)
    result = torch.zeros(weights_2d.shape[0], param_dim, dtype=mu.dtype, device=mu.device)

    for start in range(0, states_flat.numel(), chunk_size):
        end = min(start + chunk_size, states_flat.numel())
        chunk_weights = weights_2d[:, start:end]
        active = chunk_weights.abs().sum(dim=1) > 0
        if not active.any():
            continue

        active_indices = active.nonzero(as_tuple=False).squeeze(1)
        chunk_weights = chunk_weights[active]
        states_chunk = states_flat[start:end]
        actions_chunk = actions_flat[start:end]
        mu_chunk = mu_flat[start:end]
        batch = end - start
        groups = chunk_weights.shape[0]

        time = torch.full(
            (batch, 1),
            t / max(1, time_scale),
            dtype=mu.dtype,
            device=mu.device,
        )
        z = torch.cat([time, mu_chunk], dim=-1)
        pre1 = torch.nn.functional.linear(z, linear1.weight, linear1.bias)
        hidden1 = torch.tanh(pre1)
        pre2 = torch.nn.functional.linear(hidden1, linear2.weight, linear2.bias)
        hidden2 = torch.tanh(pre2)
        logits = torch.nn.functional.linear(hidden2, linear3.weight, linear3.bias).reshape(
            batch,
            n_states,
            n_actions,
        )
        probs = torch.softmax(logits, dim=-1)

        delta_logits = torch.zeros_like(logits)
        rows = torch.arange(batch, device=mu.device)
        delta_logits[rows, states_chunk] = -probs[rows, states_chunk]
        delta_logits[rows, states_chunk, actions_chunk] += 1.0
        delta_out = chunk_weights.unsqueeze(-1) * delta_logits.reshape(batch, n_states * n_actions).unsqueeze(0)

        grad_w3 = torch.einsum("gso,sh->goh", delta_out, hidden2)
        grad_b3 = delta_out.sum(dim=1)
        delta_hidden2 = torch.einsum("gso,oh->gsh", delta_out, linear3.weight)
        delta_pre2 = delta_hidden2 * (1.0 - hidden2.square()).unsqueeze(0)

        grad_w2 = torch.einsum("gsh,si->ghi", delta_pre2, hidden1)
        grad_b2 = delta_pre2.sum(dim=1)
        delta_hidden1 = torch.einsum("gsh,hi->gsi", delta_pre2, linear2.weight)
        delta_pre1 = delta_hidden1 * (1.0 - hidden1.square()).unsqueeze(0)

        grad_w1 = torch.einsum("gsh,si->ghi", delta_pre1, z)
        grad_b1 = delta_pre1.sum(dim=1)
        flat = torch.cat(
            [
                grad_w1.reshape(groups, -1),
                grad_b1.reshape(groups, -1),
                grad_w2.reshape(groups, -1),
                grad_b2.reshape(groups, -1),
                grad_w3.reshape(groups, -1),
                grad_b3.reshape(groups, -1),
            ],
            dim=1,
        )
        result[active_indices] += flat.detach()

    if group_shape:
        return result.reshape(*group_shape, param_dim)
    return result.squeeze(0)
