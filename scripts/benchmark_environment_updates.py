from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfc.algorithms import LogitsPerturbedMFREINFORCE, SimplexPerturbedMFREINFORCE
from mfc.environments import (
    CybersecurityConfig,
    CybersecurityMFC,
    CybersecurityPolicy,
    DistributionPlanningConfig,
    DistributionPlanningMFC,
    DistributionPlanningPolicy,
    TwoStateConfig,
    TwoStateMFC,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label: str, fn, repeats: int = 3) -> None:
    fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    print(f"{label}: {elapsed:.4f}s/update")


def benchmark_twostate(device: torch.device, dtype: torch.dtype, skip_logits: bool) -> None:
    config = TwoStateConfig(device=device, dtype=dtype)
    env = TwoStateMFC(config)
    theta = torch.zeros(env.n_states, dtype=dtype, device=device)
    mu0 = torch.tensor([0.2, 0.8], dtype=dtype, device=device)
    flow = env.exact_population_flow(theta, mu0, config.T)
    simplex = SimplexPerturbedMFREINFORCE(env)
    logits = LogitsPerturbedMFREINFORCE(env)

    timed("two-state simplex B=200 n=10 T=2", lambda: simplex.complete_gradient_estimate(theta, flow, 0.2, 200, 10))
    if not skip_logits:
        timed(
            "two-state logits N=200 n=10 T=2",
            lambda: logits.gradient_estimate(theta, mu0, 0.2, 200, 10, 1, horizon=config.T, mu_flow=flow),
        )


def benchmark_cybersecurity(device: torch.device, dtype: torch.dtype, full_budget: bool, skip_logits: bool) -> None:
    config = CybersecurityConfig(device=device, dtype=dtype, hidden_units=32, T_train=3)
    env = CybersecurityMFC(config)
    policy = CybersecurityPolicy(config)
    mu0 = torch.full((config.n_states,), 1.0 / config.n_states, dtype=dtype, device=device)
    flow = env.exact_population_flow(policy, mu0, config.T_train)
    simplex = SimplexPerturbedMFREINFORCE(env)
    logits = LogitsPerturbedMFREINFORCE(env)

    timed("cybersecurity simplex B=200 n=10 T=3", lambda: simplex.complete_gradient_estimate(policy, flow, 0.2, 200, 10))
    if not skip_logits:
        timed(
            "cybersecurity logits N=200 n=10 T=3",
            lambda: logits.gradient_estimate(policy, mu0, 0.2, 200, 10, 1, horizon=config.T_train, mu_flow=flow),
        )
    if full_budget:
        timed(
            "cybersecurity simplex equal-budget B=3675 n=525 T=3",
            lambda: simplex.complete_gradient_estimate(policy, flow, 0.2, 3675, 525),
            repeats=1,
        )


def benchmark_distribution(device: torch.device, dtype: torch.dtype, full_budget: bool, skip_logits: bool) -> None:
    config = DistributionPlanningConfig(device=device, dtype=dtype, hidden_units=256, T=5)
    env = DistributionPlanningMFC(config)
    policy = DistributionPlanningPolicy(config)
    mu0 = torch.full((config.n_states,), 1.0 / config.n_states, dtype=dtype, device=device)
    flow = env.exact_population_flow(policy, mu0, config.T)
    simplex = SimplexPerturbedMFREINFORCE(env)
    logits = LogitsPerturbedMFREINFORCE(env)

    timed("distribution simplex B=200 n=10 T=5", lambda: simplex.complete_gradient_estimate(policy, flow, 0.2, 200, 10))
    if not skip_logits:
        timed(
            "distribution logits N=200 n=10 T=5",
            lambda: logits.gradient_estimate(policy, mu0, 0.2, 200, 10, 1, horizon=config.T, mu_flow=flow),
        )
    if full_budget:
        timed(
            "distribution simplex equal-budget B=5425 n=775 T=5",
            lambda: simplex.complete_gradient_estimate(policy, flow, 0.2, 5425, 775),
            repeats=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one optimized MF-REINFORCE update per environment.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--full-budget", action="store_true", help="Also run the large equal-budget simplex checks.")
    parser.add_argument("--skip-logits", action="store_true", help="Skip logits benchmarks and run only simplex checks.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"device={device} dtype={dtype}")
    benchmark_twostate(device, dtype, args.skip_logits)
    benchmark_cybersecurity(device, dtype, args.full_budget, args.skip_logits)
    benchmark_distribution(device, dtype, args.full_budget, args.skip_logits)


if __name__ == "__main__":
    main()
