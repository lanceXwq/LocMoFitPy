"""`run_locmofit`, the entry point that MATLAB and Python call, and `fit_problem` (as LocMoFit.jl's LocMoFit.jl)."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .bfgs import bfgs
from .loss import FitStatic, fit_data, loss_value, parameters
from .models import Composite, get_model
from .parameters import ParameterMap, apply_init
from .priors import ParamToFit, prior_terms, specification
from .start import axis_vector, estimate_start, start_names


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
    prior_terms: tuple  # each prior that is not flat as (kind, names, family) (priors.prior_terms)
    prior_numbers: tuple  # and its numbers
    dtype: Any


def fit_problem(
    model,
    locs: ArrayLike,
    loc_precisions: ArrayLike,
    *,
    params: Mapping[str, ParamToFit] | None = None,
    spacing: float = 3.0,
    point_params: Mapping[str, float] | None = None,
    dtype: Any = "float64",
    background: bool = True,
    fov_volume: float | None = None,
) -> tuple[Problem, jax.Array]:
    """Everything a fit needs: the `Problem` for `make_loss` and the free parameters `u0` to start from, with the starting values `params` leaves open estimated from the localizations (`start.estimate_start`). The keywords are those of `run_locmofit`."""
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
    axes = model.orientation_names()
    axis_names = {n for pair in axes for n in pair}
    # params in the optimizer's parameters: starting values, held parameters, ties, and priors
    spec = specification(model, background, params, axes)
    init, fixed, tie = dict(spec["init"]), spec["fixed"], spec["tie"]
    # what params leaves open, held and tied parameters excepted, starts from the data (LocMoFit.jl's derivations/derivation.qmd), along the starting axis of component 1 if it is given or held
    theta_a, phi_a = axes[0]
    chart = ("vx1", "vy1") if isinstance(model, Composite) else ("vx", "vy")
    given_axis = (
        theta_a in init or phi_a in init or any(k in fixed for k in chart)
    )
    axis = (
        axis_vector(float(init.get(theta_a, 0.0)), float(init.get(phi_a, 0.0)))
        if given_axis
        else None
    )
    targets = {t for t, _ in tie}

    def isopen(k: str) -> bool:
        return k not in init and k not in fixed and k not in targets

    if any(isopen(k) for k in start_names(model, locs_np.shape[1])):
        estimate = estimate_start(model, locs_np, prec_np, axis)
        init = {**{k: v for k, v in estimate.items() if isopen(k)}, **init}
    starts = tuple(
        (float(init.get(a, 0.0)), float(init.get(b, 0.0))) for a, b in axes
    )

    theta0 = model.default_parameters()
    if background:
        theta0["omega0"] = 0.0
    theta0 = apply_init(
        theta0, {k: v for k, v in init.items() if k not in axis_names}
    )
    # a background fraction not given starts at 0.1, whatever the components' shares: ω₀ = log(f₀/f₁) with f₁ = 0.9w₁ (LocMoFit.jl's derivations/derivation.qmd, @eq-softmax)
    if background and "omega0" not in init:
        lw = np.array([float(w) for w in model.log_weights(theta0)])
        theta0["omega0"] = math.log(1 / 9) + float(np.logaddexp.reduce(lw))
    pmap = ParameterMap(theta0, fixed, tie)
    u0 = pmap.free_parameters(theta0)
    theta0 = pmap.full_parameters(u0)  # with the ties applied
    # the values the points are generated from: point_params in place of the starting values, with the ties applied
    points_from = theta0
    if point_params:
        points_from = apply_init(theta0, dict(point_params))
        pm = ParameterMap(points_from, fixed, tie)
        points_from = pm.full_parameters(pm.free_parameters(points_from))
    unit = model.unit_model(
        points_from, starts, spacing=spacing, dtype=dtype.type
    )

    terms, numbers = prior_terms(spec["priors"], unit)
    sizes = [len(points) for points, _ in unit]
    ends = np.cumsum(sizes)
    ranges = tuple(
        range(int(e - n), int(e)) for e, n in zip(ends, sizes, strict=True)
    )
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
        prior_terms=terms,
        prior_numbers=numbers,
        dtype=dtype,
    )
    return problem, jnp.asarray(u0, dtype=dtype)


def run_locmofit(
    model,
    locs: ArrayLike,
    loc_precisions: ArrayLike,
    *,
    params: Mapping[str, ParamToFit] | None = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    spacing: float = 3.0,
    point_params: Mapping[str, float] | None = None,
    dtype: Any = "float64",
    background: bool = True,
    fov_volume: float | None = None,
    history: bool = False,
) -> dict[str, Any]:
    """Fit `model` to one site's localizations by maximizing the posterior of LocMoFit.jl's derivations/derivation.qmd: a mixture of the structure and a uniform background, with a prior on every quantity (@eq-full-posterior), by BFGS with a backtracking line search (`bfgs`).

    - `model`: a model (`Ring()`, `Composite(Ring(), SphericalCap())`, …) or a name (`"Ring"`, `"SphericalCap"`, `"DoubleRing"`, `"SpCapRing"`, case-insensitive, or an alias: `"spcap"`, `"npc"`).
    - `locs`, `loc_precisions`: (N, D) positions and per-axis precisions (standard deviations), in nm, with D ∈ {2, 3}. A 2D fit uses the first two coordinates of the model.
    - `params`: a dict of `ParamToFit`, the starting value and the prior of each quantity it names, both in the quantity's own terms; an unknown name raises a `ValueError`. A component k of a composite has its number appended to its quantities' names (`x1`, `r1`, `axis2`, …). The quantities and their default priors:
        - `x`, `y`, `z`: the position of the reference point (a ring's center, a cap's apex), in nm; for component k > 1, in component 1's frame. `Flat()`.
        - `r` (`Ring`), `s` (`SphericalCap`): the size, in nm (a cap's area is 4πs²). `LogNormal(log(50) + 1, 1)`: most probable at 50 nm, median 136 nm, 95% between 19 and 965 nm.
        - `kappa` (`SphericalCap`): the dimensionless curvature κ′ in [−1, 1], from a flat disc at 0 to a whole sphere at ±1 (a starting value inside (−1, 1)). `Flat()`. It is fitted as the cap's half-angle `alpha`, with κ′ = sin(`alpha`/2).
        - `axis`: the axis, as `(theta, phi)`, its polar angle and azimuth; for component k > 1, in component 1's frame. `Flat()`, uniform on the sphere; it takes a `Watson` or `Bingham` density. It is fitted as the coordinates `vx`, `vy` of a tangent chart centered on its starting value.
        - `sigma`: the blur, in nm, added to every precision in quadrature. `Flat()`.
        - `fractions` (with a background or several components): the fractions of the localizations from the background and from each component, as a tuple `(f0, f1, …, fK)` (without a background `(f1, …, fK)`, the components' shares), positive numbers of which only the ratios count. `Dirichlet(2, 5, …, 5)` (`Dirichlet(5)` without a background): as much as 1 background localization and 4 from each component, so that the background and every component keep some of the localizations and no component can be switched off. On the signal fraction π = 1 − f0 it acts as K·4 signal localizations and 1 background one, so that for one component it is the Beta(5, 2) density of π (mode 0.8) and for two pulls π toward 8/9; it moves a component's share w of the signal by K(alpha − 1)(1/K − w)/(n + K(alpha − 1)) for n signal localizations and alpha = 5 (with two components and 100 signal localizations, a share of 0.3 becomes 0.315).
      `Fixed()` holds a quantity at its value and `Tied("r1")` at another quantity's value throughout (LocMoFit.jl's derivations/derivation.qmd, Fixed and tied values). Which priors a fit has is part of what it is compiled for; their numbers are not, so changing them does not recompile. A starting value not given is estimated from the localizations (`start.estimate_start`; LocMoFit.jl's derivations/derivation.qmd, Starting values from the data), unless the quantity is fixed or tied: then, and wherever nothing is estimated, it starts from its default (plus a background fraction of 0.1 with a background). The estimate covers a `Ring`, a `SphericalCap` and a composite of two of them sharing an axis (such as `DoubleRing` and `SpCapRing`): in 3D the position, the axis and the shape, and for a composite the second component's distance along the axis (`z2`) and the shares; in 2D only the position. Any other model gets only the position of component 1, at the weighted center of the localizations. The blur and the background fraction start from their defaults, and an axis not estimated from (0, 0), along z. An axis given for component 1, or held, is used instead of the estimated one, and the shape is estimated along it.
    - `max_iter`, `tol`: the fit stops when no component of the loss's gradient exceeds `tol`, or after `max_iter` iterations. The default `tol`, 1e-4, leaves the parameters much closer to the optimum than their statistical error (a position within about 0.005 nm for a ring of 100 localizations, whose error is about 0.7 nm).
    - `spacing`: the distance between model points, in nm. The points are generated once, from the starting values.
    - `point_params`: values that replace the starting values only for generating the points (with the ties applied), such as `{"r": 50}`: then the number of points comes from a ring of radius 50 nm, whatever the starting radius. Only the shape parameters that set the number of points matter: a ring's `r` and a cap's `s` (`r1`, `s2`, … in a composite). Unknown names are ignored with a warning. Sites fitted with the same values have the same number of points, so starting values that differ from site to site do not add compilations (NOTES.md).
    - `dtype`: `"float64"` (default, since 2026-10-29) or `"float32"`. float64 sets `jax_enable_x64`, which is process-global and stays on for the rest of the process (a MATLAB `pyenv` session too). float32 is faster on large models (a cap fit about 1.7×) but can stop early, before the gradient reaches `tol`, with the loss up to about 1e-4 per localization above the optimum (a known issue: NOTES.md).
    - `background`: whether some localizations are background, uniform over the field of view; their share is f0 of `fractions`.
    - `fov_volume`: the volume (area, in 2D) of the field of view, in nm³ (nm²); default the localizations' bounding box, a little smaller.
    - `history`: also return the loss and the parameters at the start and after every iteration.

    Returns a dict of NumPy arrays and Python scalars (MATLAB-convertible): `parameters` (all of the optimizer's parameters, by name: `omega0`, `omega2`, … for the fractions, ωₖ = log(fₖ/f₁), `alpha` for a cap's half-angle, κ′ = sin(`alpha`/2), which may come out negative or beyond π, where the shapes repeat (`components` reports the half-angle in [0, π]), `vx`, `vy` for an axis), `signal_fraction` (1 − f0, None without a background), `weights` (the components' shares), `components` (each component in the world frame: `position`, `axis`, `theta`, `phi` and its shape: a ring's `r`; a cap's `s`, `kappa`, `curvature`, `radius`, `alpha`, `area`), `loss`, `model_points` (M, 3), `iterations`, `retcode` (`"Success"`, `"MaxIters"` or `"Failure"`) and `history` (None, or `{"losses", "parameters"}` with one entry per iteration and the start).
    """
    problem, u0 = fit_problem(
        model,
        locs,
        loc_precisions,
        params=params,
        spacing=spacing,
        point_params=point_params,
        dtype=dtype,
        background=background,
        fov_volume=fov_volume,
    )
    data, static = fit_data(problem, pad=True)
    result, points = compiled_fit(static, max_iter, float(tol), history)(
        u0, data
    )
    return fit_result(problem, static, result, points, history)


@functools.cache
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
        for (s, _), unit in zip(static.blocks, problem.unit_model, strict=True)
    ]
    return {
        "parameters": theta,
        "signal_fraction": weights.sum()
        / (weights.sum() + math.exp(theta["omega0"] - lw.max()))
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
