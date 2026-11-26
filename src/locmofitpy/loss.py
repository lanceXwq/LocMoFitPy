"""The likelihood of LocMoFit.jl's derivations/derivations.qmd: each localization's density is a mixture of the structure (the mean of Gaussians over each component's model points, the components weighted) and a uniform background."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .priors import log_prior


def log_likelihood(
    locs,
    loc_prec,
    points,
    ranges,
    sigma,
    logw,
    b,
    background,
    loc_weight=None,
    point_valid=None,
    counts=None,
):
    """Σₙ log(pₙ + eᵇ) over the localizations (the rows of the (N, D) `locs`, with precisions `loc_prec`), or Σₙ log pₙ without a background.

    pₙ is the structure's density at localization n: for each component k, whose model points are the rows `ranges[k]` of `points` (of which the first D columns are used), the mean of the Gaussians around its points with per-axis variances `loc_prec² + sigma²`; the components weighted by `exp(logw)`. In a fit, `logw` are the log fractions of the components and `b` the log of the background's density times its fraction (LocMoFit.jl's derivations/derivations.qmd, @eq-mixture). The squared distances are formed from differences (not expanded), so the result keeps its precision far from the origin, also in float32.

    For padded arrays (as `run_locmofit` compiles them, so that sites of similar sizes share one compilation), `loc_weight` (N,) is 1 for a localization and 0 for padding, `point_valid` (M,) marks the real points, and `counts[k]` is component k's number of real points; padding then contributes nothing. Its gradient is in closed form (`_ll_bwd`; LocMoFit.jl's derivations/derivations.qmd, the gradient of the log-likelihood).
    """
    dtype = locs.dtype
    if loc_weight is None:
        loc_weight = jnp.ones(locs.shape[0], dtype)
    if point_valid is None:
        point_valid = jnp.ones(points.shape[0], bool)
    if counts is None:
        counts = jnp.asarray([r.stop - r.start for r in ranges], dtype)
    return _ll(
        locs,
        loc_prec,
        points,
        sigma,
        jnp.asarray(logw, dtype),
        jnp.asarray(b, dtype),
        loc_weight,
        point_valid,
        counts,
        tuple(ranges),
        background,
    )


def _neg_q(locs, h, points, point_valid):
    """-Σ_d h_nd (x_nd - y_md)², the exponent of each localization's Gaussian around each point (N, M), one dimension at a time (faster than an (N, M, D) array); -∞ for padding points."""
    D = locs.shape[1]
    neg_q = -sum(
        h[:, d, None] * (locs[:, d, None] - points[None, :, d]) ** 2
        for d in range(D)
    )
    return jnp.where(point_valid[None, :], neg_q, -jnp.inf)


def _ll_forward(
    locs,
    loc_prec,
    points,
    sigma,
    logw,
    b,
    loc_weight,
    point_valid,
    counts,
    ranges,
    background,
):
    v = loc_prec**2 + sigma**2  # (N, D)
    h = 1 / (2 * v)
    c = (
        -jnp.sum(jnp.log(2 * math.pi * v), axis=1) / 2
    )  # the Gaussians' normalization
    neg_q = _neg_q(locs, h, points, point_valid)
    lse = jnp.stack(
        [logsumexp(neg_q[:, r.start : r.stop], axis=1) for r in ranges], axis=1
    )  # (N, K)
    t = logw[None, :] - jnp.log(counts)[None, :] + lse
    S = logsumexp(t, axis=1)
    L = jnp.logaddexp(c + S, b) if background else c + S
    total = jnp.sum(jnp.where(loc_weight > 0, L, 0))
    return total, (h, c, lse, t, S, L)


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def _ll(
    locs,
    loc_prec,
    points,
    sigma,
    logw,
    b,
    loc_weight,
    point_valid,
    counts,
    ranges,
    background,
):
    return _ll_forward(
        locs,
        loc_prec,
        points,
        sigma,
        logw,
        b,
        loc_weight,
        point_valid,
        counts,
        ranges,
        background,
    )[0]


def _ll_fwd(
    locs,
    loc_prec,
    points,
    sigma,
    logw,
    b,
    loc_weight,
    point_valid,
    counts,
    ranges,
    background,
):
    total, res = _ll_forward(
        locs,
        loc_prec,
        points,
        sigma,
        logw,
        b,
        loc_weight,
        point_valid,
        counts,
        ranges,
        background,
    )
    return total, (
        locs,
        loc_prec,
        points,
        sigma,
        loc_weight,
        point_valid,
        counts,
        res,
    )


def _ll_bwd(ranges, background, saved, gbar):
    """The gradient with respect to the points, σ, the components' log fractions and the background's log term, from the shares of LocMoFit.jl's derivations/derivations.qmd: rₙ (the localization is signal), q_nk (it comes from component k) and the weights of the points within each component. The exponents are recomputed rather than kept from the forward pass, and the gradient with respect to the points is two matrix products on coordinates centered at the localizations' mean (exact up to rounding of that size); the one with respect to the variances keeps the differences."""
    (
        locs,
        loc_prec,
        points,
        sigma,
        loc_weight,
        point_valid,
        counts,
        (h, c, lse, t, S, L),
    ) = saved
    D = locs.shape[1]
    wt = (loc_weight > 0).astype(locs.dtype)
    rho = jnp.exp(c + S - L) if background else jnp.ones_like(S)
    r = wt * rho  # (N,)
    g = r[:, None] * jnp.exp(
        t - S[:, None]
    )  # (N, K): ∂L/∂ each component's log-sum-exp
    neg_q = _neg_q(locs, h, points, point_valid)
    A = jnp.concatenate(
        [
            g[:, k, None]
            * jnp.exp(neg_q[:, rg.start : rg.stop] - lse[:, k, None])
            for k, rg in enumerate(ranges)
        ],
        axis=1,
    )  # (N, M)
    center = jnp.mean(locs, axis=0)
    xc, yc = locs - center, points[:, :D] - center
    d_points = 2 * (
        A.T @ (h * xc) - yc * (A.T @ h)
    )  # Σₙ A_nm 2h_nd (x_nd − y_md)
    d_points = jnp.concatenate(
        [
            d_points,
            jnp.zeros((points.shape[0], points.shape[1] - D), points.dtype),
        ],
        axis=1,
    )
    d_h = -jnp.stack(
        [
            jnp.sum(A * (locs[:, d, None] - points[None, :, d]) ** 2, axis=1)
            for d in range(D)
        ],
        axis=1,
    )
    d_v = (
        -r[:, None] * h - 2 * h**2 * d_h
    )  # v = s² + σ² enters c (∂c/∂v = -h) and h (∂h/∂v = -2h²)
    d_sigma = 2 * sigma * jnp.sum(d_v)
    d_logw = jnp.sum(g, axis=0)
    d_b = jnp.sum(wt * (1 - rho)) if background else jnp.zeros_like(sigma)
    zero = jnp.zeros_like
    return (
        zero(locs),
        zero(loc_prec),
        gbar * d_points,
        gbar * d_sigma,
        gbar * d_logw,
        gbar * d_b,
        zero(loc_weight),
        None,
        zero(counts),
    )


_ll.defvjp(_ll_fwd, _ll_bwd)


class FitData(NamedTuple):
    """The arrays a compiled fit takes as arguments (see `fit_data`)."""

    locs: Any
    loc_prec: Any
    loc_weight: Any
    n: Any  # the number of (real) localizations
    unit_model: tuple  # one (points, chart) per component
    point_valid: Any
    counts: Any
    base: Any
    S: Any
    log_volume: Any
    priors: tuple  # the numbers of each prior term (priors.prior_terms), as data, so that changing them does not recompile


@dataclass(frozen=True)
class FitStatic:
    """What a compiled fit is specialized for: the model's structure and the layout of the arrays. Equal `FitStatic`s share one compilation (`pipeline.compiled_fit`)."""

    model_key: tuple
    names: tuple[str, ...]
    blocks: tuple[
        tuple[int, int], ...
    ]  # each component's rows among the (padded) points
    background: bool
    priors: (
        tuple  # each prior term as (kind, names, family) (priors.prior_terms)
    )
    masked: bool
    model: Any = field(compare=False, hash=False)


def model_key(model) -> tuple:
    """The model's structure: its type and, for a composite, its components' types (the defaults do not enter the loss)."""
    if hasattr(model, "components") and isinstance(model.components, tuple):
        return (
            type(model).__name__,
            tuple(type(c).__name__ for c in model.components),
        )
    return (type(model).__name__,)


def parameters(u, data: FitData, static: FitStatic) -> dict:
    """All parameters θ = base + S·u, by name, as JAX scalars (traceable, unlike `ParameterMap.full_parameters`)."""
    theta = data.base + data.S @ u
    return {k: theta[i] for i, k in enumerate(static.names)}


def loss_value(u, data: FitData, static: FitStatic):
    """Minus the log posterior of the parameters, per localization (LocMoFit.jl's derivations/derivations.qmd, @eq-full-posterior): the log-likelihood plus the log density of every prior that is not flat."""
    theta = parameters(u, data, static)
    model = static.model
    points = model.model_points(data.unit_model, theta)
    # the log fractions of the mixture, log softmax(ω₀, 0, ω₂, …) (LocMoFit.jl's derivations/derivations.qmd, @eq-logfractions)
    lw = jnp.stack(model.log_weights(theta))
    if static.background:
        lse = jnp.logaddexp(theta["omega0"], logsumexp(lw))
        b = theta["omega0"] - lse - data.log_volume
    else:
        lse = logsumexp(lw)
        b = 0.0
    prior = log_prior(static.priors, data.priors, theta)
    ranges = tuple(range(s, e) for s, e in static.blocks)
    masks = (
        dict(
            loc_weight=data.loc_weight,
            point_valid=data.point_valid,
            counts=data.counts,
        )
        if static.masked
        else {}
    )
    ll = log_likelihood(
        data.locs,
        data.loc_prec,
        points,
        ranges,
        theta["sigma"],
        lw - lse,
        b,
        static.background,
        **masks,
    )
    return -(ll + prior) / data.n


def bucket(n: int) -> int:
    """`n` rounded up to a size with at most 4 significant bits (…, 112, 120, 128, 144, …), so that padding wastes at most 1/8 and sizes from n to 2n share 8 compilations."""
    step = 2 ** max(0, n.bit_length() - 4)
    return -(-n // step) * step


def fit_data(problem, pad: bool) -> tuple[FitData, FitStatic]:
    """The arrays and the static description of `problem` for `loss_value`; with `pad`, the localizations and each component's points are padded to `bucket` sizes and masked."""
    dtype = problem.dtype
    locs, prec = np.asarray(problem.locs), np.asarray(problem.loc_prec)
    N = len(locs)
    units, valid, blocks, counts = [], [], [], []
    start = 0
    for points, chart in problem.unit_model:
        M = len(points)
        Mp = bucket(M) if pad else M
        padded = (
            np.concatenate([points, np.repeat(points[:1], Mp - M, axis=0)])
            if Mp > M
            else points
        )
        units.append((jnp.asarray(padded, dtype), jnp.asarray(chart, dtype)))
        valid.append(np.arange(Mp) < M)
        blocks.append((start, start + Mp))
        counts.append(M)
        start += Mp
    Np = bucket(N) if pad else N
    if Np > N:  # padding rows repeat the first localization, with weight 0
        locs = np.concatenate([locs, np.repeat(locs[:1], Np - N, axis=0)])
        prec = np.concatenate([prec, np.repeat(prec[:1], Np - N, axis=0)])
    pmap = problem.parameter_map
    data = FitData(
        locs=jnp.asarray(locs, dtype),
        loc_prec=jnp.asarray(prec, dtype),
        loc_weight=jnp.asarray(np.arange(Np) < N, dtype),
        n=jnp.asarray(N, dtype),
        unit_model=tuple(units),
        point_valid=jnp.asarray(np.concatenate(valid)),
        counts=jnp.asarray(counts, dtype),
        base=jnp.asarray(pmap.base, dtype),
        S=jnp.asarray(pmap.S, dtype),
        log_volume=jnp.asarray(problem.log_volume, dtype),
        priors=tuple(jnp.asarray(p, dtype) for p in problem.prior_numbers),
    )
    static = FitStatic(
        model_key=model_key(problem.model),
        names=pmap.names,
        blocks=tuple(blocks),
        background=problem.background,
        priors=problem.prior_terms,
        masked=pad,
        model=problem.model,
    )
    return data, static


def make_loss(problem, pad: bool = False):
    """The objective of a fit, as a function of the free parameters `u`: minus the log posterior of the parameters, per localization (`problem` from `fit_problem`). With `pad`, it is evaluated on the padded arrays that `run_locmofit` compiles."""
    data, static = fit_data(problem, pad)
    return lambda u: loss_value(u, data, static)
