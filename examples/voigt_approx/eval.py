#!/usr/bin/env python3
"""Evaluate the Voigt profile approximation: accuracy vs scipy ground truth + speed.

The Voigt profile is:
    V(x; sigma, gamma) = Re[w(z)] / (sigma * sqrt(2*pi))
where z = (x + i*gamma) / (sigma * sqrt(2)) and w is the Faddeeva function.

Ground truth is computed via scipy.special.wofz.
The approximation in experiment/voigt.py must use only numpy and math.
"""

import json
import math
import time

import numpy as np
from scipy.special import wofz

from experiment.voigt import approx_voigt_batch
from experiment.config import N_POINTS, N_REPEATS, METHOD


# Test cases: (sigma, gamma) pairs covering different regimes
TEST_CASES = [
    (1.0, 0.1),   # near-Gaussian
    (0.1, 1.0),   # near-Lorentzian
    (0.5, 0.5),   # equal mix
    (0.2, 0.8),   # Lorentzian-dominated
    (0.8, 0.2),   # Gaussian-dominated
]

X_RANGE = (-5.0, 5.0)


def voigt_ground_truth(x_array, sigma, gamma):
    """Exact Voigt profile via scipy Faddeeva function."""
    z = (x_array + 1j * gamma) / (sigma * math.sqrt(2))
    return np.real(wofz(z)) / (sigma * math.sqrt(2 * math.pi))


def time_function(fn, *args, n_repeats=N_REPEATS):
    """Return best-of-n elapsed time in seconds."""
    fn(*args)  # warmup
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def evaluate():
    x_grid = np.linspace(X_RANGE[0], X_RANGE[1], N_POINTS)

    total_mse = 0.0
    total_ref_time = 0.0
    total_approx_time = 0.0
    details = []

    for sigma, gamma in TEST_CASES:
        # Ground truth
        y_true = voigt_ground_truth(x_grid, sigma, gamma)

        # Approximation output
        try:
            y_approx = np.asarray(approx_voigt_batch(x_grid.copy(), sigma, gamma), dtype=np.float64).flatten()
        except Exception as e:
            return {"score": 0.0, "description": f"Error in approx_voigt_batch(sigma={sigma}, gamma={gamma}): {e}"}

        if y_approx.shape != y_true.shape:
            return {"score": 0.0, "description": f"Wrong output shape: got {y_approx.shape}, expected {y_true.shape}"}

        mse = float(np.mean((y_approx - y_true) ** 2))
        total_mse += mse

        # Timing
        t_ref = time_function(voigt_ground_truth, x_grid, sigma, gamma)
        t_approx = time_function(approx_voigt_batch, x_grid, sigma, gamma)
        total_ref_time += t_ref
        total_approx_time += t_approx

        speed_ratio = t_ref / max(t_approx, 1e-12)
        details.append(f"s={sigma},g={gamma}: MSE={mse:.2e}, speed={speed_ratio:.2f}x")

    avg_mse = total_mse / len(TEST_CASES)
    avg_ref_time_ms = (total_ref_time / len(TEST_CASES)) * 1000
    avg_approx_time_ms = (total_approx_time / len(TEST_CASES)) * 1000

    mse_penalty = math.exp(-avg_mse * 50000)
    # Speed score: how much faster than scipy's wofz (>=1x is perfect)
    speed_ratio = total_ref_time / max(total_approx_time, 1e-12)
    normalized_speed = max(0.0, min(1.0, speed_ratio))

    score = normalized_speed * mse_penalty

    description = (
        f"MSE={avg_mse:.2e}, "
        f"ref={avg_ref_time_ms:.2f}ms, "
        f"approx={avg_approx_time_ms:.2f}ms, "
        f"speed={speed_ratio:.2f}x, "
        f"acc={mse_penalty:.4f} | "
        + "; ".join(details)
    )

    return {"score": round(score, 6), "description": description}


if __name__ == "__main__":
    result = evaluate()
    result_out = {"score": result["score"], "description": result["description"]}
    print(json.dumps(result_out))
