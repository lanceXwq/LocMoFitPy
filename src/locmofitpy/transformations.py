"""Rotations and placement (as LocMoFit.jl's transformations.jl). Points are rows, (M, 3)."""

import jax.numpy as jnp
import numpy as np


def rotmat(theta: float, phi: float, dtype=np.float64) -> np.ndarray:
    """The minimal rotation taking z to the axis at polar angle `theta` and azimuth `phi`: the frame of a starting axis."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    t = 1 - ct
    return np.array(
        [
            [1 - t * cp * cp, -t * sp * cp, st * cp],
            [-t * sp * cp, 1 - t * sp * sp, st * sp],
            [-st * cp, -st * sp, ct],
        ],
        dtype=dtype,
    )


def chart_rotation(R0, vx, vy):
    """The rotation that orients a model whose axis has tangent-chart coordinates `(vx, vy)`.

    The chart's frame `R0` (from `rotmat` of the starting axis) turns z to its center; the axis is `R0·m` with `m = normalize((vx, vy, 1))`, and the rotation is `R0` times the minimal rotation taking z to `m`. `m₃ > 0` for every `(vx, vy)`, so that minimal rotation never meets its singularity (at `m = -z`), and the chart reaches every axis within 90° of its center (LocMoFit.jl's derivations/derivation.qmd).
    """
    r = 1 / jnp.sqrt(1 + vx**2 + vy**2)
    m1, m2, m3 = vx * r, vy * r, r
    k = 1 / (1 + m3)
    M = jnp.array(
        [
            [1 - k * m1**2, -k * m1 * m2, m1],
            [-k * m1 * m2, 1 - k * m2**2, m2],
            [-m1, -m2, m3],
        ]
    )
    return jnp.asarray(R0, M.dtype) @ M


def place(X, R, x, y, z):
    """The points X (M, 3) rotated by R and shifted by (x, y, z)."""
    return X @ R.T + jnp.stack([x, y, z])
