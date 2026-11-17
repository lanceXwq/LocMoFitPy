"""Priors on every quantity a fit estimates, and `ParamToFit`, which gives each quantity its starting value and its prior (as LocMoFit.jl's priors.jl; its derivations/derivation.qmd, Priors on every parameter).

A prior is a density of a quantity of the structure (a position, a size, a curvature, an axis, the fractions of the mixture), whatever coordinates the optimizer uses for it, and the loss adds its log density with no Jacobian (@eq-full-posterior). Which priors a fit has is part of what it is compiled for (`loss.FitStatic`); their numbers are data (`loss.FitData`), so that changing them does not recompile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


class Prior:
    """A prior on one quantity of a fit, or on a group of them (an axis, the fractions), given in `ParamToFit`. Its log density, up to a constant, is added to the log-likelihood. The densities: `Flat`, `Normal`, `LogNormal`, `Dirichlet`, `Watson`, `Bingham`; and the point masses `Fixed` and `Tied`, which hold a quantity at its starting value or at another quantity's value."""


@dataclass(frozen=True)
class Flat(Prior):
    """A constant density: no prior term. It is proper on a bounded quantity (a cap's `kappa` in [−1, 1], an axis on the sphere, the fractions) and improper on a position or a size, which the localizations must then determine."""


@dataclass(frozen=True)
class Normal(Prior):
    """The normal density of a position (nm), a size (nm; of its magnitude, so that `Normal(0, sd)` on `sigma` is a half-normal) or a cap's `kappa` (restricted to [−1, 1])."""

    mean: float
    sd: float

    def __post_init__(self):
        if not self.sd > 0:
            raise ValueError(f"Normal: sd must be positive, got {self.sd}")


@dataclass(frozen=True)
class LogNormal(Prior):
    """The log-normal density of a size (nm): its logarithm is normal with mean `mu` and standard deviation `sigma`, so its median is exp(`mu`) and it spreads by a factor exp(`sigma`) (@eq-lognormal). It vanishes at 0, so a size with this prior cannot shrink to a point. Its most probable value, which a fit's maximum favors, is exp(`mu` − `sigma`²). The default of a ring's `r` and a cap's `s` is `LogNormal(log(50) + 1, 1)`: most probable at 50 nm, median 136 nm, 95% of it between 19 and 965 nm."""

    mu: float
    sigma: float

    def __post_init__(self):
        if not self.sigma > 0:
            raise ValueError(
                f"LogNormal: sigma must be positive, got {self.sigma}"
            )


@dataclass(frozen=True, init=False)
class Dirichlet(Prior):
    """The Dirichlet density of the fractions of a fit's mixture, ∝ ∏ₖ fₖ^(alphaₖ − 1) (LocMoFit.jl's derivations/derivation.qmd, @eq-fraction-prior): `Dirichlet(alpha)`, the same for every fraction, or `Dirichlet(alpha0, alpha1, …, alphaK)`, with a background its fraction f₀ and each component's f₁, …, f_K (the parameters in that order); without one, the components' shares. Each must be at least 1 (below 1 the density grows without bound as a fraction goes to 0). It acts like alphaₖ − 1 extra localizations from the background and from each component (@eq-fraction-map), and keeps every fraction above 0, so that no component can be switched off. On the signal fraction π = 1 − f₀ and the shares it is a Beta density of π, with parameters Σₖ₌₁ᴷ(alphaₖ − 1) + 1 and alpha0, and the Dirichlet density of the shares with alpha1, …, alphaK (@eq-fraction-split). The default is `Dirichlet(2, 5, …, 5)` with a background (for one component the Beta(5, 2) density of π, mode 0.8; for two, as 8 extra signal localizations and 1 background one, a pull toward π = 8/9 that grows with the number of components), and `Dirichlet(5)` without."""

    alpha: float | tuple

    def __init__(self, *alpha):
        if not alpha:
            raise ValueError("Dirichlet: alpha is missing")
        if not all(a >= 1 for a in alpha):
            raise ValueError(
                f"Dirichlet: every alpha must be at least 1, got {alpha}: below 1 the density grows without bound as a fraction goes to 0"
            )
        a = float(alpha[0]) if len(alpha) == 1 else tuple(map(float, alpha))
        object.__setattr__(self, "alpha", a)


@dataclass(frozen=True)
class Watson(Prior):
    """The Watson density of an axis on the sphere, ∝ exp(`kappa` (μ·n)²), with μ the unit vector at polar angle `theta` and azimuth `phi` (@eq-watson). It gives n and −n the same density, as the likelihood does. `kappa > 0` keeps the axis near ±μ, `kappa < 0` near the great circle perpendicular to μ (with `theta = 0`, an axis near the xy plane), and `kappa = 0` is uniform. For component k > 1 of a composite, μ is in component 1's frame, as the axis is."""

    theta: float
    phi: float
    kappa: float


@dataclass(frozen=True, init=False)
class Bingham(Prior):
    """The Bingham density of an axis on the sphere, ∝ exp(Σⱼ kappaⱼ (μⱼ·n)²): the product of the Watson densities `terms`, whose exponents it adds (LocMoFit.jl's derivations/derivation.qmd, @eq-bingham). With perpendicular μ's it holds the axis to a different spread in each direction: `Bingham(Watson(0, 0, -1000), Watson(math.pi / 2, -math.pi / 2, 2.6))` keeps an axis near the xy plane (its tilt out of it with SD ≈ 1/√(2·1000) rad ≈ 1.3°) and near ±y within it (its turn with SD ≈ 1/√(2·2.6) rad ≈ 25°). It gives n and −n the same density and is proper for every kappa. For component k > 1 of a composite, the μ's are in component 1's frame, as the axis is."""

    terms: tuple

    def __init__(self, *terms):
        if not terms or not all(isinstance(t, Watson) for t in terms):
            raise ValueError(
                f"a Bingham density is given as one or more Watson terms, got {terms!r}"
            )
        object.__setattr__(self, "terms", tuple(terms))


@dataclass(frozen=True)
class Fixed(Prior):
    """A point mass at the quantity's starting value: the quantity is not fitted. Without a value it is held at its default."""


@dataclass(frozen=True)
class Tied(Prior):
    """A point mass at another quantity's value, such as `Tied("r1")` for `r2`: the quantity takes the source's value throughout, and only the source's prior applies. Both must be of the same kind (positions, sizes or curvatures)."""

    source: str


@dataclass(frozen=True)
class ParamToFit:
    """What is known about one quantity of a fit before it: its starting `value` (None to estimate it from the localizations, or else start from its default) and its `prior` (a `Prior`; None for the quantity's default prior). Both are in the quantity's own terms: a position or a size in nm, a cap's `kappa` (κ′), an axis as `(theta, phi)`, the fractions as a tuple. `ParamToFit(prior)` with a prior alone leaves the value open. `run_locmofit` lists the quantities and their default priors."""

    value: Any = None
    prior: Prior | None = None

    def __post_init__(self):
        if isinstance(self.value, Prior):
            if self.prior is not None:
                raise ValueError(
                    f"ParamToFit takes a value and a prior, got two priors: {self.value}, {self.prior}"
                )
            object.__setattr__(self, "prior", self.value)
            object.__setattr__(self, "value", None)
        if self.prior is not None and not isinstance(self.prior, Prior):
            raise ValueError(f"not a prior: {self.prior!r}")


@dataclass(frozen=True)
class Quantity:
    """A quantity priors are stated in, computed from the optimizer's parameters `names`: a `"location"` (a position, nm), a `"size"` (the magnitude of a parameter, which the likelihood is even in), a `"curvature"` (κ′ = sign(s)·sin(α/2), from the half-angle α and s: the cap of scale −s is the cap of scale s with the opposite curvature), an `"axis"` (of `component`, from its chart coordinates vx, vy) or the `"fractions"` of the mixture (their logarithms, log softmax(ω₀, 0, ω₂, …) with a background, whose parameter `omega0` comes first in `names`, and log softmax(0, ω₂, …) without; @eq-logfractions)."""

    kind: str
    names: tuple[str, ...]
    component: int = 1


SIZE_PRIOR = LogNormal(math.log(50.0) + 1, 1.0)
# the default Dirichlet's alpha for the background's fraction and for each component's
BACKGROUND_ALPHA = 2.0
COMPONENT_ALPHA = 5.0


def nfractions(q: Quantity) -> int:
    """The number of fractions of a `"fractions"` quantity: one more than its parameters (component 1's is the reference)."""
    return len(q.names) + 1


# which priors each kind of quantity takes (Flat and Fixed fit every kind)
_SUPPORTS = {
    "location": (Normal, Tied),
    "size": (Normal, LogNormal, Tied),
    "curvature": (Normal, Tied),
    "axis": (Watson, Bingham),
    "fractions": (Dirichlet,),
}
_DESCRIBE = {
    "location": "position",
    "size": "size",
    "curvature": "curvature",
    "axis": "axis",
    "fractions": "set of fractions",
}


def supports(q: Quantity, prior: Prior) -> bool:
    return isinstance(prior, (Flat, Fixed, *_SUPPORTS[q.kind]))


def start_values(q: Quantity, value, axes) -> dict[str, float]:
    """A quantity's value as starting values of the parameters; `axes` are the names that set each component's starting axis (`orientation_names`)."""
    if q.kind in ("location", "size"):
        return {q.names[0]: float(value)}
    if q.kind == "curvature":
        if not abs(value) < 1:
            raise ValueError(f"kappa must lie in (-1, 1), got {value}")
        return {q.names[0]: 2 * math.asin(value)}
    if q.kind == "axis":
        if np.ndim(value) != 1 or len(value) != 2:
            raise ValueError(
                f"an axis is given as (theta, phi), got {value!r}"
            )
        theta, phi = axes[q.component - 1]
        return {theta: float(value[0]), phi: float(value[1])}
    # the fractions, as ωₖ = log(fₖ/f₁), with f₁ the fraction of component 1 (@eq-softmax)
    n = nfractions(q)
    if np.ndim(value) != 1 or len(value) != n or not all(v > 0 for v in value):
        raise ValueError(
            f"the fractions are given as {n} positive numbers, got {value!r}"
        )
    ref = 1 if q.names[0] == "omega0" else 0
    others = [i for i in range(n) if i != ref]
    return {
        name: math.log(value[i]) - math.log(value[ref])
        for name, i in zip(q.names, others)
    }


def specification(model, background: bool, params, axes) -> dict:
    """What `params` (a dict of `ParamToFit`) asks of a fit of `model`, in the optimizer's parameters: the starting values (`init`, with each starting axis under its `axes` names), the parameters held (`frozen`), the ties (`tie`, `(target, source)` pairs), and the priors as `(quantity, prior)` pairs (`priors`; flat ones left out). Raises a `ValueError` for an unknown name, a prior that does not fit its quantity, or a value outside its quantity's range."""
    table = model.quantities(background)
    params = dict(params or {})
    unknown = tuple(k for k in params if k not in table)
    if unknown:
        raise ValueError(
            f"unknown quantities {unknown}; the quantities of this fit are {tuple(table)}"
        )
    init: dict[str, float] = {}
    frozen: list[str] = []
    tie: list[tuple[str, str]] = []
    priors: list[tuple[Quantity, Prior]] = []
    for name, (q, default) in table.items():
        spec = params.get(name, ParamToFit())
        if not isinstance(spec, ParamToFit):
            raise ValueError(
                f"params[{name!r}] must be a ParamToFit, got {spec!r}"
            )
        prior = default if spec.prior is None else spec.prior
        if not supports(q, prior):
            raise ValueError(
                f"{name} is a {_DESCRIBE[q.kind]}, which does not take a {type(prior).__name__} prior"
            )
        if spec.value is not None:
            init.update(start_values(q, spec.value, axes))
        if isinstance(prior, Fixed):
            frozen.extend(q.names)
        elif isinstance(prior, Tied):
            if prior.source not in table or prior.source == name:
                raise ValueError(
                    f"{name} is tied to {prior.source}, which is not another quantity of this fit"
                )
            source = table[prior.source][0]
            if source.kind != q.kind:
                raise ValueError(
                    f"{name}, a {_DESCRIBE[q.kind]}, is tied to {prior.source}, a {_DESCRIBE[source.kind]}"
                )
            tie.append((q.names[0], source.names[0]))
        elif not isinstance(prior, Flat):
            priors.append((q, prior))
    return dict(init=init, frozen=tuple(frozen), tie=tuple(tie), priors=priors)


def prior_terms(priors, unit_model) -> tuple[tuple, tuple]:
    """The priors of `specification` as the loss evaluates them: a static description of each term, `(kind, names, family)`, which the fit is compiled for, and its numbers, which the fit takes as data. A Watson or Bingham density becomes the quadratic form mᵀA′m in the chart's coordinates, with A′ = R₀ᵀAR₀ for the axis n = R₀m and A = Σⱼ κⱼμⱼμⱼᵀ, fixed for the whole fit; its numbers are A′'s six distinct entries a₁₁, a₂₂, a₃₃, a₁₂, a₁₃, a₂₃."""
    static, numbers = [], []
    for q, p in priors:
        values: tuple[float, ...]
        if isinstance(p, Normal):
            family, values = "normal", (p.mean, p.sd)
        elif isinstance(p, LogNormal):
            family, values = "lognormal", (p.mu, p.sigma)
        elif isinstance(p, Dirichlet):
            n = nfractions(q)
            if isinstance(p.alpha, tuple) and len(p.alpha) != n:
                raise ValueError(
                    f"the Dirichlet prior has {len(p.alpha)} alphas for {n} fractions"
                )
            family = "dirichlet"
            values = p.alpha if isinstance(p.alpha, tuple) else (p.alpha,) * n
        else:  # Watson or Bingham
            R0 = np.asarray(unit_model[q.component - 1][1], dtype=float)
            A = np.zeros((3, 3))
            for t in p.terms if isinstance(p, Bingham) else (p,):
                mu = R0.T @ np.array(
                    [
                        math.sin(t.theta) * math.cos(t.phi),
                        math.sin(t.theta) * math.sin(t.phi),
                        math.cos(t.theta),
                    ]
                )
                A += t.kappa * np.outer(mu, mu)
            family = "quadratic"
            values = (A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2])
        static.append((q.kind, q.names, family))
        numbers.append(tuple(float(v) for v in values))
    return tuple(static), tuple(numbers)


def quantity(kind: str, names: tuple[str, ...], theta: dict):
    """The quantity `kind` computed from the parameters `theta` (JAX scalars)."""
    if kind == "location":
        return theta[names[0]]
    if kind == "size":
        return jnp.abs(theta[names[0]])
    if kind == "curvature":
        return jnp.sign(theta[names[1]]) * jnp.sin(theta[names[0]] / 2)
    if kind == "axis":
        vx, vy = theta[names[0]], theta[names[1]]
        r = 1 / jnp.sqrt(1 + vx**2 + vy**2)
        return jnp.stack([vx * r, vy * r, r])
    # the fractions: component 1's ω₁ = 0 after the background's ω₀, if any
    omega = [theta[n] for n in names]
    zero = jnp.zeros_like(omega[0])
    lw = jnp.stack(
        [omega[0], zero, *omega[1:]]
        if names[0] == "omega0"
        else [zero, *omega]
    )
    return lw - logsumexp(lw)


def log_density(family: str, p, q):
    """The log density, up to a constant, of the prior `family` with numbers `p` at the quantity `q`."""
    if family == "normal":
        return -(((q - p[0]) / p[1]) ** 2) / 2
    if family == "lognormal":
        lq = jnp.log(q)
        return -lq - ((lq - p[0]) / p[1]) ** 2 / 2
    if family == "dirichlet":
        return jnp.sum((jnp.asarray(p) - 1) * q)
    # quadratic (Watson, Bingham)
    return (
        p[0] * q[0] ** 2
        + p[1] * q[1] ** 2
        + p[2] * q[2] ** 2
        + 2 * (p[3] * q[0] * q[1] + p[4] * q[0] * q[2] + p[5] * q[1] * q[2])
    )


def log_prior(static, numbers, theta):
    """The sum of the prior terms (`prior_terms`) at the parameters `theta`; 0 for none."""
    total = 0.0
    for (kind, names, family), p in zip(static, numbers):
        total = total + log_density(family, p, quantity(kind, names, theta))
    return total
