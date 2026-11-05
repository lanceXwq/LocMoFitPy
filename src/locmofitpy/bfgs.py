"""BFGS with a backtracking line search, as Optim.jl's `BFGS(linesearch=BackTracking())` that LocMoFit.jl uses, jitted.

The same algorithm, so that both packages reach the same optimum by the same path up to rounding:
- the inverse Hessian starts as the identity and gets the BFGS update whenever dx·dg > 0; if the direction −H·g is not a descent direction, H is reset to the identity;
- the line search is LineSearches.jl's `BackTracking` (c₁ = 1e-4, step factors in [0.1, 0.5], a quadratic and then cubic interpolation), starting from a step of 1; if it fails, H is reset and it is tried once more;
- the fit stops when the largest gradient component is at most `tol`, when a step does not change the parameters, when the loss is unchanged in two successive iterations, when the line search fails, or after `max_iter` iterations.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

C1, RHO_HI, RHO_LO, LS_ITERATIONS = 1e-4, 0.5, 0.1, 1000


class BFGSResult(NamedTuple):
    x: jax.Array
    f: jax.Array
    g: jax.Array
    iterations: jax.Array
    converged: jax.Array  # by the gradient, the step or the loss
    failed: jax.Array  # the line search failed, or the new point is not finite
    history_f: jax.Array  # the loss at the start and after every iteration (empty without history)
    history_x: jax.Array


def _linesearch(value_and_grad, x, s, f0, dphi0):
    """LineSearches.jl's BackTracking (order 3) from a step of 1. Returns the step, whether the search succeeded, and the value and gradient there. Every trial point gets its gradient along with its value: the search nearly always accepts its first point, which then needs no second evaluation (NOTES.md)."""
    dtype = x.dtype
    phi = lambda a: value_and_grad(x + a * s)  # noqa: E731
    ok = jnp.isfinite(f0) & jnp.isfinite(dphi0) & (dphi0 <= 0)
    one = jnp.ones((), dtype)
    iterfinitemax = -math.log2(float(jnp.finfo(dtype).eps))

    # halve the step until the value is finite
    def finite_cond(c):
        a, (fa, _), it = c
        return ~jnp.isfinite(fa) & (it < iterfinitemax)

    def finite_body(c):
        a, _, it = c
        a = a / 2
        return a, phi(a), it + 1

    a2, (f1, g1), _ = lax.while_loop(
        finite_cond, finite_body, (one, phi(one), 0)
    )
    ok = ok & jnp.isfinite(f1)

    # backtrack until the sufficient decrease (Armijo) condition holds
    def cond(c):
        a1, a2, fx0, fx1, g1, it, failed = c
        return (fx1 > f0 + C1 * a2 * dphi0) & ~failed & ok

    def body(c):
        a1, a2, fx0, fx1, g1, it, failed = c
        it = it + 1
        quadratic = -(dphi0 * a2**2) / (2 * (fx1 - f0 - dphi0 * a2))
        div = 1 / (a1**2 * a2**2 * (a2 - a1))
        ca = (
            a1**2 * (fx1 - f0 - dphi0 * a2) - a2**2 * (fx0 - f0 - dphi0 * a1)
        ) * div
        cb = (
            -(a1**3) * (fx1 - f0 - dphi0 * a2)
            + a2**3 * (fx0 - f0 - dphi0 * a1)
        ) * div
        disc = jnp.maximum(cb**2 - 3 * ca * dphi0, 0)
        cubic = jnp.where(
            ca == 0, -dphi0 / (2 * cb), -dphi0 / (cb + jnp.sqrt(disc))
        )
        a_tmp = jnp.where(it == 1, quadratic, cubic)
        a1 = a2
        a_tmp = jnp.fmin(
            a_tmp, a2 * RHO_HI
        )  # avoid too small reductions (NaN-ignoring, as NaNMath)
        a_new = jnp.fmax(a_tmp, a2 * RHO_LO)  # avoid too big reductions

        # halve the step while the value is not finite
        def nf_cond(cc):
            a, (fa, _), i = cc
            return ~jnp.isfinite(fa) & (i < LS_ITERATIONS)

        def nf_body(cc):
            a, _, i = cc
            a = a / 2
            return a, phi(a), i + 1

        a_new, (f_new, g_new), it = lax.while_loop(
            nf_cond, nf_body, (a_new, phi(a_new), it)
        )
        failed = failed | (it > LS_ITERATIONS) | ~jnp.isfinite(f_new)
        return a1, a_new, fx1, f_new, g_new, it, failed

    _, a2, _, f2, g2, _, failed = lax.while_loop(
        cond, body, (one, a2, f0, f1, g1, 0, jnp.array(False))
    )
    return a2, ok & ~failed & (a2 > 0), f2, g2


def bfgs(
    fun, x0, *, max_iter: int = 200, tol: float = 1e-4, history: bool = False
) -> BFGSResult:
    """Minimize the jittable scalar function `fun` from `x0` (a vector); see the module's docstring for the algorithm."""
    dtype = x0.dtype
    n = x0.shape[0]
    eye = jnp.eye(n, dtype=dtype)
    gtol = jnp.asarray(tol, dtype)
    value_and_grad = jax.value_and_grad(fun)
    nhist = max_iter + 1 if history else 0

    f0, g0 = value_and_grad(x0)
    hf = jnp.zeros((nhist,), dtype)
    hx = jnp.zeros((nhist, n), dtype)
    if history:
        hf, hx = hf.at[0].set(f0), hx.at[0].set(x0)

    def cond(c):
        x, f, g, H, it, converged, failed, fcount, hf, hx = c
        return ~converged & ~failed & (it < max_iter)

    def body(c):
        x, f, g, H, it, converged, failed, fcount, hf, hx = c
        s = -H @ g
        reset = g @ s >= 0
        H = jnp.where(reset, eye, H)
        s = jnp.where(reset, -g, s)
        alpha, ok, f_new, g_new = _linesearch(value_and_grad, x, s, f, g @ s)

        # on failure, reset the inverse Hessian and try once more
        def retry(_):
            s2 = -g
            a2, ok2, f2, g2 = _linesearch(value_and_grad, x, s2, f, g @ s2)
            return eye, s2, a2, ok2, f2, g2

        H, s, alpha, ok, f_new, g_new = lax.cond(
            ok, lambda _: (H, s, alpha, ok, f_new, g_new), retry, None
        )
        x_new = x + alpha * s
        accept = ok & jnp.isfinite(f_new) & jnp.all(jnp.isfinite(g_new))

        dx = x_new - x
        dg = g_new - g
        dxdg = dx @ dg
        u = H @ dg
        c1 = (dxdg + dg @ u) / dxdg**2
        c2 = 1 / dxdg
        H_new = (
            H
            + c1 * jnp.outer(dx, dx)
            - c2 * (jnp.outer(u, dx) + jnp.outer(dx, u))
        )
        H_new = jnp.where(dxdg > 0, H_new, H)

        x_conv = jnp.max(jnp.abs(dx)) <= 0
        fcount_new = jnp.where(jnp.abs(f_new - f) <= 0, fcount + 1, 0)
        g_conv = jnp.max(jnp.abs(g_new)) <= gtol
        conv = x_conv | g_conv | (fcount_new > 1)

        # a failed step leaves the state as it was, as Optim does
        x, f, g, H = (
            jnp.where(accept, x_new, x),
            jnp.where(accept, f_new, f),
            jnp.where(accept, g_new, g),
            jnp.where(accept, H_new, H),
        )
        it_new = jnp.where(accept, it + 1, it)
        if history:
            hf = hf.at[it_new].set(f)
            hx = hx.at[it_new].set(x)
        return (
            x,
            f,
            g,
            H,
            it_new,
            accept & conv,
            ~accept,
            jnp.where(accept, fcount_new, fcount),
            hf,
            hx,
        )

    init = (
        x0,
        f0,
        g0,
        eye,
        jnp.asarray(0),
        jnp.max(jnp.abs(g0)) <= gtol,
        jnp.array(False),
        jnp.asarray(0),
        hf,
        hx,
    )
    x, f, g, _, it, converged, failed, _, hf, hx = lax.while_loop(
        cond, body, init
    )
    return BFGSResult(x, f, g, it, converged, failed, hf, hx)
