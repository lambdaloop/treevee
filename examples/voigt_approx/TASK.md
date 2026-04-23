# Problem: Fast Approximation of the Voigt Profile

## Background

The **Voigt profile** is one of the most important functions in spectroscopy, plasma physics, and astrophysics. It describes the shape of a spectral line broadened by two physical mechanisms simultaneously:
- **Gaussian broadening** (from thermal motion / Doppler effect), width σ
- **Lorentzian broadening** (from natural linewidth / pressure), width γ

It's defined as the convolution of a Gaussian and Lorentzian:

```
V(x; σ, γ) = Re[w(z)] / (σ√(2π))

where z = (x + iγ) / (σ√2)
      w(z) = exp(-z²) erfc(-iz)   [Faddeeva function]
```

Computing `w(z)` (the Faddeeva function) exactly is expensive — it requires evaluating a complex integral. This is the bottleneck in spectral fitting pipelines that call this function millions of times. Fast approximation is an active research area.

## Objective

Implement `approx_voigt_batch(x_array, sigma, gamma)` that approximates the Voigt profile as **accurately and quickly** as possible. Your score combines accuracy (vs scipy ground truth) and speed (vs scipy's wofz).

## Files You Can Modify

- `experiment/voigt.py` — the approximation implementation
- `experiment/config.py` — configuration (N_POINTS, N_REPEATS, METHOD)

You may **NOT** modify:
- `eval.py` — the evaluation harness

## Interface

Your function must have this exact signature:

```python
def approx_voigt_batch(x_array, sigma, gamma):
    """Approximate the Voigt profile over an array of x values.

    Args:
        x_array: numpy array of x positions (shape: [N])
        sigma: Gaussian width parameter (float > 0)
        gamma: Lorentzian width parameter (float > 0)

    Returns:
        numpy array of Voigt profile values (shape: [N], dtype float64)
    """
```

## Test Cases

The evaluation runs over 5 `(σ, γ)` parameter pairs:

| sigma | gamma | Regime |
|-------|-------|--------|
| 1.0 | 0.1 | near-Gaussian |
| 0.1 | 1.0 | near-Lorentzian |
| 0.5 | 0.5 | equal mix |
| 0.2 | 0.8 | Lorentzian-dominated |
| 0.8 | 0.2 | Gaussian-dominated |

Each case evaluates on a grid of 10,000 x points over `[-5, 5]`.

## Scoring

```
mse_penalty    = exp(-avg_MSE * 50000)
speed_ratio    = scipy_time / approx_time   (> 1 means faster than scipy)
normalized_speed = clamp(speed_ratio, 0, 1)
score          = normalized_speed * mse_penalty
```

The goal is to be **both accurate** (low MSE) and **fast** (faster than or comparable to scipy's wofz). Current baseline: score ~0.1-0.3.

## Constraints

1. **Libraries**: Only `numpy` and Python `math` are allowed. No `scipy`, no `numba`, no `ctypes`, no external packages.
2. **No calling scipy**: You cannot import or call `scipy.special.wofz` or any scipy function.
3. **Correctness**: Your output must be a numpy float64 array of the same shape as `x_array`.

## Hints

The Voigt profile has a rich literature of approximation methods:

- **Pseudo-Voigt** (Thompson, Cox & Hastings 1987): weighted sum of Gaussian + Lorentzian. Fast but ~1% error.
- **Humlíček's rational approximation** (1982): 4-term rational function in the complex plane. ~0.02% error, very fast.
- **Weideman's algorithm** (1994): 16/32-term Fourier series. Highly accurate.
- **Zaghloul & Ali** (2011): hybrid rational approximation, sub-ppm accuracy.
- **Chebyshev expansion** on subintervals of x: piecewise accuracy.
- **Padé approximants**: rational functions that can outperform Taylor series.
- **Lookup table + interpolation**: precompute on a grid, interpolate at runtime.
- **Exploiting symmetry**: V(x) = V(-x), compute only for x ≥ 0.
- **Vectorized numpy**: avoid Python loops — process entire x_array at once.

Key mathematical relationships that may help:
```
w(z) = exp(-z²) * erfc(-iz)
Re[w(x + iy)] for real x, y > 0 is what we need
The profile decays as ~1/x² for large |x| (Lorentzian tail)
The profile is even: V(-x) = V(x)
```
