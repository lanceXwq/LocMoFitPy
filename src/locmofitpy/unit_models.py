"""Unit point sets the models are built from (as LocMoFit.jl's `unit models.jl`), as NumPy arrays of shape (M, k)."""

import math

import numpy as np

GOLDEN_ANGLE = (3 - math.sqrt(5)) * math.pi


def unit_ring(dtype, npoints: int) -> np.ndarray:
    """`npoints` equally spaced points on the unit circle in the xy plane, (M, 3)."""
    phi = np.linspace(0, 2 * math.pi * (1 - 1 / npoints), npoints).astype(
        dtype
    )
    return np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)


def unit_sphere(dtype, npoints: int) -> np.ndarray:
    """`npoints` equal-area points on the unit sphere (a Fibonacci lattice), (M, 3)."""
    i = np.arange(npoints).astype(dtype)
    z = 1 - (i + dtype(0.5)) * dtype(2 / npoints)
    r = np.sqrt(1 - z**2)
    phi = i * dtype(GOLDEN_ANGLE)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def unit_cap(dtype, npoints: int) -> np.ndarray:
    """The unit sphere's points as (ρ̂₁, ρ̂₂, h): the direction of each point around the axis and its depth below the north pole, h = 1 - z ∈ (0, 2), which the constant-area cap needs (SphericalCap)."""
    X = unit_sphere(dtype, npoints)
    rho = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    return np.stack([X[:, 0] / rho, X[:, 1] / rho, 1 - X[:, 2]], axis=1)
