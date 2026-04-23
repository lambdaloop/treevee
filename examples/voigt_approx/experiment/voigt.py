import math
import numpy as np


def approx_voigt(x, sigma, gamma):
    """Approximate the Voigt profile at a single x value.

    Args:
        x: position (float)
        sigma: Gaussian width parameter (float > 0)
        gamma: Lorentzian width parameter (float > 0)

    Returns:
        Approximate Voigt profile value (float)
    """
    # Baseline: simple pseudo-Voigt (Thompson, Cox & Hastings, 1987)
    # eta blends Lorentzian and Gaussian by a heuristic mixing parameter
    f = sigma + gamma
    eta = 1.36603 * (gamma / f) - 0.47719 * (gamma / f) ** 2 + 0.11116 * (gamma / f) ** 3
    f_G = math.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    f_L = (gamma / math.pi) / (x ** 2 + gamma ** 2)
    return eta * f_L + (1 - eta) * f_G


def approx_voigt_batch(x_array, sigma, gamma):
    """Approximate the Voigt profile over an array of x values.

    Args:
        x_array: numpy array of x positions
        sigma: Gaussian width parameter (float > 0)
        gamma: Lorentzian width parameter (float > 0)

    Returns:
        numpy array of approximate Voigt profile values
    """
    return np.array([approx_voigt(float(x), sigma, gamma) for x in x_array])
