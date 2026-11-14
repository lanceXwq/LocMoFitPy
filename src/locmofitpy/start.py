"""Starting values estimated from the localizations (as LocMoFit.jl's start.jl; its derivations/derivation.qmd, Starting values from the data).

The estimate runs once per site, before the fit, in NumPy (float64 whatever the fit's dtype). The direction search scores all candidate axes of a set at once, as (N, K) arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import Composite, Ring, SphericalCap
from .unit_models import GOLDEN_ANGLE

START_COARSE = 100  # candidate axes spread over the hemisphere
START_FINE = 64  # candidate axes around the best coarse one,
START_CONE = math.radians(25)  # within this angle of it
# |κ′| of a starting cap, which keeps it off the whole sphere (alpha = ±π), where the fit's slope in alpha vanishes
START_MAX_KAPPA = 0.95


def estimate_start(model, locs, loc_precisions, axis=None) -> dict[str, float]:
    """Starting values for `model` estimated from one site's localizations ((N, D) positions and precisions, nm), in the optimizer's parameters (with the starting axis as `theta`, `phi`): in 3D the position, the starting axis (`theta`, `phi`) and the shape of a `Ring` or a `SphericalCap`, and for a composite of two of them sharing an axis also the second component's shape and distance along the axis (`z2`) and the shares (`omega2`); in 2D only the position. `axis`, a unit vector or None, is a starting axis given by the user, along which the rest is estimated; with None the axis is estimated too. Any other model gets only the position of its component 1, at the weighted center of the localizations, so that its fit starts at the site rather than at the origin; the rest starts from its defaults. Values that come out undefined (too few localizations) are left out."""
    locs = np.asarray(locs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if isinstance(model, (Ring, SphericalCap)):
            f = site_frame(locs, loc_precisions)
            if f is None:
                return {}
            if locs.shape[1] == 2:
                return finite_values({"x": f.center[0], "y": f.center[1]})
            n = principal_axis(f) if axis is None else np.asarray(axis, float)
            return finite_values(simple_start(model, f, n, axis is None))
        if not two_simple_parts(model):
            return position_start(model, locs, loc_precisions)
        parts = model.components
        f = site_frame(locs, loc_precisions)
        if f is None:
            return {}
        if locs.shape[1] == 2:
            return finite_values({"x1": f.center[0], "y1": f.center[1]})
        if axis is None:
            n, first_low = search_axis(parts, f)
        else:
            n = np.asarray(axis, float)
            first_low = bool(axis_scores(parts, f, n[:, None])[1][0])
        # the groups along the chosen axis
        t = AxisTerms(f, n[:, None])
        w, rho, rho2 = t.w[:, 0], np.sqrt(t.rho2[:, 0]), t.rho2[:, 0]
        nlow = int(t.nlow[0])
        low, high = t.order[:nlow, 0], t.order[nlow:, 0]
        # a cap's parabola fits a ring's group as well (all at one height and distance), so a ring and a cap are told apart by the spread of the distances from the axis: least, relative to their median, in the ring's group
        if sum(isinstance(p, Ring) for p in parts) == 1:
            ring_low = relative_spread(rho[low]) < relative_spread(rho[high])
            first_low = isinstance(parts[0], Ring) == ring_low
        g1, g2 = (low, high) if first_low else (high, low)
        # an estimated axis points from component 1 to component 2, so that z2 > 0 as in the presets
        if axis is None and np.median(w[g2]) < np.median(w[g1]):
            n, w = -n, -w
        t1, shape1 = part_start(parts[0], f, w, rho, rho2, g1, n)
        t2, shape2 = part_start(parts[1], f, w, rho, rho2, g2, n)
        theta1, phi1 = axis_angles(n)
        start = {
            "x1": t1[0],
            "y1": t1[1],
            "z1": t1[2],
            "theta1": theta1,
            "phi1": phi1,
            **suffixed(shape1, 1),
            "z2": float(np.dot(t2 - t1, n)),
            **suffixed(shape2, 2),
            "omega2": math.log(len(g2) / len(g1)),
        }
        return finite_values(start)


def position_start(model, locs, loc_precisions) -> dict[str, float]:
    """The position of component 1 at the weighted center of the localizations."""
    f = site_frame(locs, loc_precisions)
    if f is None:
        return {}
    return finite_values(
        dict(zip(position_names(model, locs.shape[1]), f.center))
    )


def position_names(model, D: int) -> tuple[str, ...]:
    names = (
        ("x1", "y1", "z1") if isinstance(model, Composite) else ("x", "y", "z")
    )
    return names[:D]


def two_simple_parts(model) -> bool:
    return (
        isinstance(model, Composite)
        and len(model.components) == 2
        and all(isinstance(p, (Ring, SphericalCap)) for p in model.components)
    )


def start_names(model, D: int) -> tuple[str, ...]:
    """The names `estimate_start` gives `model` in D dimensions, so that `fit_problem` can skip it when they are all given."""
    if isinstance(model, (Ring, SphericalCap)):
        if D == 2:
            return ("x", "y")
        return ("x", "y", "z", *model.shape_parameters(), "theta", "phi")
    if not two_simple_parts(model):
        return position_names(model, D)
    if D == 2:
        return ("x1", "y1")
    s1, s2 = (
        tuple(suffixed(c.shape_parameters(), k))
        for k, c in enumerate(model.components, start=1)
    )
    return (
        "x1",
        "y1",
        "z1",
        "theta1",
        "phi1",
        *s1,
        "z2",
        *s2,
        "omega2",
    )


def suffixed(shape: dict, k: int) -> dict:
    return {f"{n}{k}": v for n, v in shape.items()}


def finite_values(v: dict) -> dict[str, float]:
    return {k: float(x) for k, x in v.items() if math.isfinite(x)}


def axis_vector(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def axis_angles(n) -> tuple[float, float]:
    return (
        math.acos(min(max(float(n[2]), -1.0), 1.0)),
        math.atan2(float(n[1]), float(n[0])),
    )


def relative_spread(v) -> float:
    """The standard deviation over the median."""
    return float(np.sqrt(np.mean((v - np.mean(v)) ** 2)) / np.median(v))


@dataclass(frozen=True)
class Frame:
    """The localizations kept, relative to their weighted mean `center`: `rel` (N, D), their precision variances `var` and weights `c`."""

    rel: np.ndarray
    var: np.ndarray
    c: np.ndarray
    center: np.ndarray


def site_frame(locs, loc_precisions) -> Frame | None:
    """The localizations within twice the median distance from their coordinate-wise median, weighted by the inverse of each one's summed precision variance (normalized to sum to 1). None with fewer than 4 left."""
    X = np.asarray(locs, dtype=float)
    V = np.asarray(loc_precisions, dtype=float) ** 2
    med = np.median(X, axis=0)
    dist = np.sqrt(((X - med) ** 2).sum(axis=1))
    keep = dist < 2 * np.median(dist)
    if keep.sum() < 4:
        return None
    X, V = X[keep], V[keep]
    c = 1 / V.sum(axis=1)
    c /= c.sum()
    center = c @ X
    return Frame(rel=X - center, var=V, c=c, center=center)


def principal_axis(f: Frame) -> np.ndarray:
    """The eigenvector of the weighted covariance, less the mean precision variance, with the smallest eigenvalue: the axis of a ring or a cap."""
    C = (f.rel * f.c[:, None]).T @ f.rel - np.diag(f.c @ f.var)
    return np.linalg.eigh(C)[1][:, 0]


def heights(rel, n):
    """Each localization's height along n and its distance from the axis through the center."""
    w = rel @ n
    rho = np.sqrt(np.maximum((rel**2).sum(axis=1) - w**2, 0))
    return w, rho


def simple_start(model, f: Frame, n, free_axis: bool) -> dict:
    if isinstance(model, Ring):
        if free_axis and n[2] < 0:
            n = -n
        _, rho = heights(f.rel, n)
        theta, phi = axis_angles(n)
        return {
            "x": f.center[0],
            "y": f.center[1],
            "z": f.center[2],
            "r": np.median(rho),
            "theta": theta,
            "phi": phi,
        }
    w, rho = heights(f.rel, n)
    a, b = parabola(w, rho**2)
    # an estimated axis is turned so that the cap bends toward −n (κ′ ≥ 0); turned around, a cap is the same with −κ′
    if free_axis and b > 0:
        n, a, b = -n, -a, -b
    apex = f.center + a * n
    s, alpha = cap_shape(f.rel - a * n, f.var, f.c, b)
    theta, phi = axis_angles(n)
    return {
        "x": apex[0],
        "y": apex[1],
        "z": apex[2],
        "s": s,
        "alpha": alpha,
        "theta": theta,
        "phi": phi,
    }


def parabola(w, x):
    """The least-squares fit w ≈ a + b·x, as (a, b)."""
    k = len(w)
    Sx, Sy, Sxx, Sxy = x.sum(), w.sum(), (x * x).sum(), x @ w
    b = (k * Sxy - Sx * Sy) / (k * Sxx - Sx**2)
    return (Sy - b * Sx) / k, b


def cap_shape(d, var, c, b):
    """A cap's size from E|x − apex|² = 2s² + Σ_d s_d² (`d`: the localizations relative to the apex), averaged with the weights c, and its curvature −2b."""
    m = c @ ((d**2).sum(axis=1) - var.sum(axis=1))
    s = np.sqrt(np.maximum(m / (2 * c.sum()), 1.0))
    alpha = 2 * np.arcsin(
        np.clip(-2 * b * s, -START_MAX_KAPPA, START_MAX_KAPPA)
    )
    return s, alpha


def part_start(part, f: Frame, w, rho, rho2, G, n):
    """The reference point (a ring's center, a cap's apex) and the shape of a component from its group G (indices), along the axis n."""
    if isinstance(part, Ring):
        return f.center + np.median(w[G]) * n, {"r": np.median(rho[G])}
    a, b = parabola(w[G], rho2[G])
    s, alpha = cap_shape(f.rel[G] - a * n, f.var[G], f.c[G], b)
    return f.center + a * n, {"s": s, "alpha": alpha}


class AxisTerms:
    """For the candidate axes `C` (3, K), each localization's height `w` along a candidate, squared distance `rho2` from it and expected squared precision errors `along` it and `across` it (per direction), as (N, K); `order`, each column's localizations by height, and `nlow`, the size of the lower group: the split where the variance between the groups, k(N − k)(mean₁ − mean₂)², is largest."""

    def __init__(self, f: Frame, C):
        self.w = f.rel @ C
        self.rho2 = np.maximum((f.rel**2).sum(axis=1)[:, None] - self.w**2, 0)
        self.along = f.var @ C**2
        self.across = (f.var.sum(axis=1)[:, None] - self.along) / 2
        self.order = np.argsort(self.w, axis=0, kind="stable")
        ws = np.take_along_axis(self.w, self.order, axis=0)
        N = ws.shape[0]
        k = np.arange(1, N)[:, None]
        low = np.cumsum(ws, axis=0)[:-1]
        total = ws.sum(axis=0)
        between = k * (N - k) * (low / k - (total - low) / (N - k)) ** 2
        self.nlow = np.argmax(between, axis=0) + 1


def axis_scores(parts, f: Frame, C):
    """The score of every candidate axis (columns of C) for a composite of two `parts`: the localizations split in two along it where the variance between the groups is largest, each group scored as one part (`part_scores`), with the parts assigned to the groups in the order that scores lower; and whether the first part is in the lower group then."""
    t = AxisTerms(f, C)
    sorted_ = sorted_terms(f, t)
    low = np.arange(len(f.c))[:, None] < t.nlow[None, :]
    a = part_scores(parts[0], sorted_, low) + part_scores(
        parts[1], sorted_, ~low
    )
    b = part_scores(parts[0], sorted_, ~low) + part_scores(
        parts[1], sorted_, low
    )
    first_low = a <= b
    return np.where(first_low, a, b), first_low


def sorted_terms(f: Frame, t: AxisTerms) -> dict:
    """The terms of `t` and the weights, each column in its order by height, so that a column's lower group is its first `nlow` rows."""
    s = {
        name: np.take_along_axis(getattr(t, name), t.order, axis=0)
        for name in ("w", "rho2", "along", "across")
    }
    s["c"] = f.c[t.order]
    s["rho"] = np.sqrt(s["rho2"])
    return s


def part_scores(part, s: dict, G):
    """Each candidate's score of the group G (a (N, K) mask) as one part. A ring: the weighted squared distance to the best circle around the axis, (ρ − ρ̄)² + (w − w̄)², less the precision errors across and along the axis. A cap: the weighted least-squares residual of w ≈ a + b·ρ², less the precision error along the axis (infinite with fewer than 3 localizations or a singular fit)."""
    c = s["c"] * G
    S = c.sum(axis=0)
    if isinstance(part, Ring):
        rho_bar = (c * s["rho"]).sum(axis=0) / S
        w_bar = (c * s["w"]).sum(axis=0) / S
        terms = (
            (s["rho"] - rho_bar) ** 2
            + (s["w"] - w_bar) ** 2
            - s["across"]
            - s["along"]
        )
        return (c * terms).sum(axis=0)
    x, w = s["rho2"], s["w"]
    Sx, Sy = (c * x).sum(axis=0), (c * w).sum(axis=0)
    Sxx, Sxy = (c * x**2).sum(axis=0), (c * x * w).sum(axis=0)
    det = S * Sxx - Sx**2
    b = (S * Sxy - Sx * Sy) / det
    a = (Sy - b * Sx) / S
    score = (c * ((w - a - b * x) ** 2 - s["along"])).sum(axis=0)
    valid = (G.sum(axis=0) >= 3) & (det > np.finfo(float).eps * S * Sxx)
    return np.where(valid, score, np.inf)


def search_axis(parts, f: Frame):
    """The axis with the lowest score among directions spread over the hemisphere, then among the best one and directions around it, as (axis, first_low)."""
    n, _ = best_axis(parts, f, hemisphere(START_COARSE))
    C = np.concatenate([n[:, None], cone(n, START_CONE, START_FINE)], axis=1)
    return best_axis(parts, f, C)


def best_axis(parts, f: Frame, C):
    score, first_low = axis_scores(parts, f, C)
    score = np.where(np.isnan(score), np.inf, score)
    j = int(np.argmin(score))
    if not score[j] < np.inf:
        return C[:, 0].copy(), True
    return C[:, j].copy(), bool(first_low[j])


def hemisphere(k: int) -> np.ndarray:
    """k directions spread evenly over the hemisphere z > 0 (a Fibonacci lattice), (3, k); the score does not change with the sign of the axis."""
    i = np.arange(k)
    z = 1 - (i + 0.5) / k
    r = np.sqrt(1 - z**2)
    return np.stack(
        [r * np.cos(i * GOLDEN_ANGLE), r * np.sin(i * GOLDEN_ANGLE), z]
    )


def cone(n, alpha: float, k: int) -> np.ndarray:
    """k directions spread evenly within the angle alpha of the unit vector n, (3, k)."""
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    i = np.arange(k)
    z = 1 - (1 - math.cos(alpha)) * (i + 0.5) / k
    r = np.sqrt(1 - z**2)
    return (
        np.outer(e1, r * np.cos(i * GOLDEN_ANGLE))
        + np.outer(e2, r * np.sin(i * GOLDEN_ANGLE))
        + np.outer(n, z)
    )
