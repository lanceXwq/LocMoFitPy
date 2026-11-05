"""`run_locmofit`, the entry point that MATLAB and Python call, and `fit_problem` (as LocMoFit.jl's LocMoFit.jl)."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .bfgs import bfgs
from .loss import FitStatic, fit_data, loss_value, parameters
from .models import get_model
from .parameters import ParameterMap, apply_init, freeze_names, tie_pairs


@dataclass(frozen=True)
class Problem:
    """Everything a fit needs besides the free parameters (see `fit_problem`)."""

    model: Any
    parameter_map: ParameterMap
    unit_model: tuple
    ranges: tuple[range, ...]
    locs: jax.Array
    loc_prec: jax.Array
    background: bool
    log_volume: float
    signal_prior: tuple[float, float] | None
    dtype: Any


def fit_problem(
    model,
    locs: ArrayLike,
    loc_precisions: ArrayLike,
    *,
    init_params: Mapping[str, float] | None = None,
    freeze: str | Sequence[str] = (),
    tie: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    spacing: float = 3.0,
    dtype: Any = "float64",
    background: bool = True,
    fov_volume: float | None = None,
    signal_prior: tuple[float, float] | None = (5, 2),
) -> tuple[Problem, jax.Array]:
    """Everything a fit needs: the `Problem` for `make_loss` and the free parameters `u0` to start from. The keywords are those of `run_locmofit`."""
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    locs_np = np.asarray(locs, dtype=dtype)
    prec_np = np.asarray(loc_precisions, dtype=dtype)
    if prec_np.shape != locs_np.shape:
        raise ValueError(
            f"loc_precisions must match locs shape; got {prec_np.shape} vs {locs_np.shape}"
        )
    if locs_np.ndim != 2 or locs_np.shape[1] not in (2, 3):
        raise ValueError(f"locs must be (N, 2) or (N, 3); got {locs_np.shape}")
    if isinstance(model, str):
        model = get_model(model)

    # the starting axes, as theta, phi (thetak, phik for component k), center the orientation charts; they are not parameters
    init = dict(init_params or {})
    axes = model.orientation_names()
    axis_names = {n for pair in axes for n in pair}
    freeze = freeze_names(freeze)
    frozen_axes = tuple(k for k in freeze if k in axis_names)
    if frozen_axes:
        raise ValueError(
            f"theta and phi only set the starting axis, {frozen_axes}; to keep the orientation, freeze its chart coordinates (vx, vy, or vxk, vyk for component k)"
        )
    starts = tuple(
        (float(init.get(a, 0.0)), float(init.get(b, 0.0))) for a, b in axes
    )

    theta0 = model.default_parameters()
    if background:
        theta0["logit_prob"] = math.log(9)  # signal fraction 0.9
    theta0 = apply_init(
        theta0, {k: v for k, v in init.items() if k not in axis_names}
    )
    pmap = ParameterMap(theta0, freeze, tie_pairs(tie))
    u0 = pmap.free_parameters(theta0)
    theta0 = pmap.full_parameters(u0)  # with the ties applied
    unit = model.unit_model(theta0, starts, spacing=spacing, dtype=dtype.type)

    sizes = [len(points) for points, _ in unit]
    ends = np.cumsum(sizes)
    ranges = tuple(range(int(e - n), int(e)) for e, n in zip(ends, sizes))
    V = (
        float(np.prod(locs_np.max(axis=0) - locs_np.min(axis=0)))
        if fov_volume is None
        else float(fov_volume)
    )
    problem = Problem(
        model=model,
        parameter_map=pmap,
        unit_model=unit,
        ranges=ranges,
        locs=jnp.asarray(locs_np),
        loc_prec=jnp.asarray(prec_np),
        background=background,
        log_volume=math.log(V),
        signal_prior=None
        if signal_prior is None
        else (float(signal_prior[0]), float(signal_prior[1])),
        dtype=dtype,
    )
    return problem, jnp.asarray(u0, dtype=dtype)


def run_locmofit(
    model,
    locs: ArrayLike,
    loc_precisions: ArrayLike,
    *,
    init_params: Mapping[str, float] | None = None,
    freeze: str | Sequence[str] = (),
    tie: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    spacing: float = 3.0,
    dtype: Any = "float64",
    background: bool = True,
    fov_volume: float | None = None,
    signal_prior: tuple[float, float] | None = (5, 2),
    history: bool = False,
) -> dict[str, Any]:
    """Fit `model` to one site's localizations by maximizing the posterior of LocMoFit.jl's derivations/derivation.qmd: a mixture of the structure and a uniform background, with a Beta prior on the signal fraction, by BFGS with a backtracking line search (`bfgs`).

    - `model`: a model (`Ring()`, `Composite(Ring(), SphericalCap())`, …) or a name (`"Ring"`, `"SphericalCap"`, `"DoubleRing"`, `"SpCapRing"`, case-insensitive, or an alias: `"spcap"`, `"npc"`).
    - `locs`, `loc_precisions`: (N, D) positions and per-axis precisions (standard deviations), in nm, with D ∈ {2, 3}. A 2D fit uses the first two coordinates of the model.
    - `init_params`: starting values that replace the defaults (plus `logit_prob` with a background); unknown names are ignored with a warning. The starting axis is given as `theta`, `phi` (polar angle and azimuth; `thetak`, `phik` for component k, in component 1's frame for k > 1, default 0): the orientation is fitted as the coordinates `vx`, `vy` of a tangent chart centered on it, which start at 0.
    - `freeze`: names of the parameters kept at their starting values (for the orientation, `vx` and `vy`).
    - `tie`: `{target: source}` (or pairs), such as `{"r2": "r1"}`: the target takes the source's value throughout.
    - `max_iter`, `tol`: the fit stops when no component of the loss's gradient exceeds `tol` (default 1e-4, reachable in float32), or after `max_iter` iterations.
    - `spacing`: the distance between model points, in nm. The points are generated once, from the starting values.
    - `dtype`: `"float64"` (default, since 2026-10-29) or `"float32"`. float64 sets `jax_enable_x64`, which is process-global and stays on for the rest of the process (a MATLAB `pyenv` session too). float32 is faster on large models (a cap fit about 1.7×) but can stop early, before the gradient reaches `tol`, with the loss up to about 1e-4 per localization above the optimum (a known issue: NOTES.md).
    - `background`: whether some localizations are background, uniform over the field of view; their share is 1 − logistic(`logit_prob`), fitted from 0.1.
    - `fov_volume`: the volume (area, in 2D) of the field of view, in nm³ (nm²); default the localizations' bounding box, a little smaller.
    - `signal_prior`: `(a, b)` of the Beta prior on the signal fraction, or None. The default, (5, 2), has its mode at 0.8 and weighs as much as 5 localizations.
    - `history`: also return the loss and the parameters at the start and after every iteration.

    Returns a dict of NumPy arrays and Python scalars (MATLAB-convertible): `parameters` (all of them, by name), `signal_fraction` (None without a background), `weights` (the components' shares), `components` (each component in the world frame: `position`, `axis`, `theta`, `phi` and its shape: a ring's `r`; a cap's `s`, `kappa`, `curvature`, `radius`, `alpha`, `area`), `loss`, `model_points` (M, 3), `iterations`, `retcode` (`"Success"`, `"MaxIters"` or `"Failure"`) and `history` (None, or `{"losses", "parameters"}` with one entry per iteration and the start).
    """
    problem, u0 = fit_problem(
        model,
        locs,
        loc_precisions,
        init_params=init_params,
        freeze=freeze,
        tie=tie,
        spacing=spacing,
        dtype=dtype,
        background=background,
        fov_volume=fov_volume,
        signal_prior=signal_prior,
    )
    data, static = fit_data(problem, pad=True)
    result, points = compiled_fit(static, max_iter, float(tol), history)(
        u0, data
    )
    return fit_result(problem, static, result, points, history)


@functools.lru_cache(maxsize=None)
def compiled_fit(static: FitStatic, max_iter: int, tol: float, history: bool):
    """The jitted fit for one model structure and array layout: `(u0, data) -> (BFGSResult, model points)`. Sites whose padded sizes agree (`loss.bucket`) reuse it, so a dataset compiles once per model and size class, not once per site."""

    def fit(u0, data):
        result = bfgs(
            lambda u: loss_value(u, data, static),
            u0,
            max_iter=max_iter,
            tol=tol,
            history=history,
        )
        theta = parameters(result.x, data, static)
        return result, static.model.model_points(data.unit_model, theta)

    return jax.jit(fit)


def fit_result(
    problem: Problem, static: FitStatic, result, points, history: bool
) -> dict[str, Any]:
    model, pmap = problem.model, problem.parameter_map
    theta = pmap.full_parameters(result.x)
    lw = np.array([float(w) for w in model.log_weights(theta)])
    weights = np.exp(lw - lw.max())
    iterations = int(result.iterations)
    retcode = (
        "Success"
        if bool(result.converged)
        else "Failure"
        if bool(result.failed)
        else "MaxIters"
    )
    hist = None
    if history:
        thetas = [
            pmap.full_parameters(u)
            for u in np.asarray(result.history_x)[: iterations + 1]
        ]
        hist = {
            "losses": np.asarray(result.history_f)[: iterations + 1],
            "parameters": {
                k: np.array([t[k] for t in thetas]) for k in pmap.names
            },
        }
    points = np.asarray(points)
    real = [
        points[s : s + (len(unit[0]))]
        for (s, _), unit in zip(static.blocks, problem.unit_model)
    ]
    return {
        "parameters": theta,
        "signal_fraction": 1 / (1 + math.exp(-theta["logit_prob"]))
        if problem.background
        else None,
        "weights": weights / weights.sum(),
        "components": model.summary(problem.unit_model, theta),
        "loss": float(result.f),
        "model_points": np.concatenate(real),
        "iterations": iterations,
        "retcode": retcode,
        "history": hist,
    }
