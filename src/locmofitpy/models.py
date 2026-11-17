"""Geometric models (as LocMoFit.jl's models.jl, ring.jl, spcap.jl and composite.jl).

A model only describes a shape: its parameters are not stored in it but in the fitted vector (`ParameterMap`). A simple model defines its shape in its own frame (its reference point, such as a ring's center or a cap's apex, at the origin and its axis along z) and is placed by a pose: the position `x`, `y`, `z` and the tangent-chart coordinates `vx`, `vy` of its axis around a starting axis. A `Composite` places its components in component 1's frame. Every model has a blur `sigma`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from .parameters import apply_init
from .priors import (
    BACKGROUND_ALPHA,
    COMPONENT_ALPHA,
    Dirichlet,
    SIZE_PRIOR,
    Flat,
    Prior,
    Quantity,
)
from .transformations import chart_rotation, place, rotmat
from .unit_models import unit_cap, unit_ring


def common_quantities(model, background: bool) -> dict:
    """The quantities every fit has besides its components': the blur `sigma` and, with a background or several components, the `fractions` of the mixture (the background's parameter `omega0`, if any, then the components' `omega2`, …), each with its default prior: `Dirichlet(2, 5, …, 5)` with a background, `Dirichlet(5)` without."""
    out: dict[str, tuple[Quantity, Prior]] = {
        "sigma": (Quantity("size", ("sigma",)), Flat())
    }
    K = model.ncomponents()
    names = tuple(f"omega{k}" for k in range(2, K + 1))
    if background:
        prior = Dirichlet(BACKGROUND_ALPHA, *(COMPONENT_ALPHA,) * K)
        out["fractions"] = (Quantity("fractions", ("omega0", *names)), prior)
    elif K > 1:
        out["fractions"] = (
            Quantity("fractions", names),
            Dirichlet(COMPONENT_ALPHA),
        )
    return out


class SimpleModel:
    """A model of one shape. Subclasses define `shape_parameters`, `unit_points`, `local_points`, `shape_quantities` and `shape_summary`."""

    def shape_parameters(self) -> dict[str, float]:
        raise NotImplementedError

    def unit_points(self, shape, *, spacing, dtype) -> np.ndarray:
        raise NotImplementedError

    def local_points(self, unit, shape):
        raise NotImplementedError

    def shape_quantities(self, suffix: str) -> dict:
        """The shape's quantities, as `{name: (quantity, default prior)}` with `suffix` appended to the names."""
        raise NotImplementedError

    def shape_summary(self, shape) -> dict:
        raise NotImplementedError

    def component_quantities(self, suffix: str, index: int) -> dict:
        """The quantities of this model as component `index` of a fit, whose parameter names end in `suffix` ("" for a model fitted alone), each with its default prior: its position, its shape and its axis."""
        return {
            **{
                f"{n}{suffix}": (
                    Quantity("location", (f"{n}{suffix}",)),
                    Flat(),
                )
                for n in ("x", "y", "z")
            },
            **self.shape_quantities(suffix),
            f"axis{suffix}": (
                Quantity("axis", (f"vx{suffix}", f"vy{suffix}"), index),
                Flat(),
            ),
        }

    def quantities(self, background: bool) -> dict:
        """The quantities `ParamToFit` gives starting values and priors for, as `{name: (quantity, default prior)}`."""
        return {
            **self.component_quantities("", 1),
            **common_quantities(self, background),
        }

    def default_parameters(self) -> dict[str, float]:
        # the blur starts at 1 nm: the likelihood depends on sigma only through sigma², so a fit started at sigma = 0 never moves it
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            **self.shape_parameters(),
            "vx": 0.0,
            "vy": 0.0,
            "sigma": 1.0,
        }

    def local_names(self) -> tuple[str, ...]:
        """The parameter names of the shape and its pose, without the blur."""
        return ("x", "y", "z", *self.shape_parameters(), "vx", "vy")

    def orientation_names(self) -> tuple[tuple[str, str], ...]:
        """The names under which each component's starting axis (the `axis` of `params`) is kept: `theta`, `phi`, or `thetak`, `phik` for component k."""
        return (("theta", "phi"),)

    def ncomponents(self) -> int:
        return 1

    def unit_model(
        self, theta: Mapping[str, float], starts, *, spacing, dtype
    ):
        """What the model is built from, made once from the starting values: the unit points and the frame of the orientation's chart, as a tuple with one `(points, chart)` per component."""
        points = self.unit_points(theta, spacing=spacing, dtype=dtype)
        return ((points, rotmat(starts[0][0], starts[0][1], dtype=dtype)),)

    def model_points(self, unit_model, theta):
        """The (M, 3) model points for the parameters `theta` (a dict of scalars)."""
        points, chart = unit_model[0]
        R = chart_rotation(chart, theta["vx"], theta["vy"])
        X = self.local_points(points, theta)
        return place(X, R, theta["x"], theta["y"], theta["z"])

    def log_weights(self, theta) -> tuple:
        return (jnp.zeros_like(theta["sigma"]),)

    def summary(self, unit_model, theta) -> list[dict]:
        """Each component in the world frame: position, axis (as a vector and as `theta`, `phi`) and shape summary."""
        R = np.asarray(
            chart_rotation(unit_model[0][1], theta["vx"], theta["vy"])
        )
        position = np.array([theta["x"], theta["y"], theta["z"]], dtype=float)
        return [
            {
                **pose_summary(position, R[:, 2]),
                **self.shape_summary(theta),
            }
        ]


def pose_summary(position, n) -> dict:
    return {
        "position": np.asarray(position, dtype=float),
        "axis": np.asarray(n, dtype=float),
        "theta": float(np.arccos(np.clip(n[2], -1, 1))),
        "phi": float(np.arctan2(n[1], n[0])),
    }


class Ring(SimpleModel):
    """A circle of radius `r` (nm), centered at its position `(x, y, z)`, in the plane normal to its axis."""

    def shape_parameters(self):
        return {"r": 50.0}

    def unit_points(self, shape, *, spacing, dtype):
        npoints = math.ceil(2 * math.pi * float(shape["r"]) / spacing)
        return unit_ring(dtype, npoints)

    def local_points(self, unit, shape):
        return shape["r"] * jnp.asarray(unit)

    def shape_quantities(self, suffix):
        # the radius, with the default prior on sizes
        r = f"r{suffix}"
        return {r: (Quantity("size", (r,)), SIZE_PRIOR)}

    def shape_summary(self, shape):
        return {"r": float(shape["r"])}


class SphericalCap(SimpleModel):
    """A spherical cap of constant area 4πs², with its apex at its position `(x, y, z)` and its outward normal along its axis.

    Its shape is fitted as `alpha`, the half-angle from the axis to the rim seen from the sphere's center, which gives the dimensionless curvature κ′ = sin(`alpha`/2) = curvature × `s` (LocMoFit.jl's derivations/derivation.qmd, @eq-cap-alpha): 0 is a flat disc of radius 2s, ±π/2 a half-sphere, ±π the whole sphere of radius s, and a negative `alpha` bends the other way. The sphere's radius is s/κ′.
    """

    def shape_parameters(self):
        return {"s": 50.0, "alpha": 0.961}  # a half-angle of 55°, κ′ ≈ 0.46

    def unit_points(self, shape, *, spacing, dtype):
        # equal-area points on the unit sphere, as many as the cap's area holds at the spacing
        npoints = math.ceil(4 * math.pi * float(shape["s"]) ** 2 / spacing**2)
        return unit_cap(dtype, npoints)

    def local_points(self, unit, shape):
        # the unit sphere's point at depth h below its north pole goes to depth s·κ′·h below the apex, at distance s·√(h(2 − κ′²h)) from the axis; each point keeps the area above it, so equal-area points stay equal-area
        unit = jnp.asarray(unit)
        kappa = jnp.sin(shape["alpha"] / 2)
        h = unit[:, 2]
        rho = jnp.sqrt(h * (2 - kappa**2 * h))
        return shape["s"] * jnp.stack(
            [rho * unit[:, 0], rho * unit[:, 1], -kappa * h], axis=1
        )

    def shape_quantities(self, suffix):
        # the scale, with the default prior on sizes, and the curvature κ′ (`kappa`, fitted as the half-angle `alpha`), uniform on [-1, 1] by default
        s = f"s{suffix}"
        return {
            s: (Quantity("size", (s,)), SIZE_PRIOR),
            f"kappa{suffix}": (
                Quantity("curvature", (f"alpha{suffix}", s)),
                Flat(),
            ),
        }

    def shape_summary(self, shape):
        s, kappa = float(shape["s"]), math.sin(float(shape["alpha"]) / 2)
        return {
            "s": s,
            "kappa": kappa,
            "curvature": kappa / s,
            "radius": s / kappa if kappa != 0 else math.inf,
            "alpha": math.acos(1 - 2 * kappa**2),
            "area": 4 * math.pi * s**2,
        }


class Composite:
    """Several simple models fitted together as one structure.

    Component 1 is placed in the world by its pose; every other component is placed by its pose in component 1's frame, so that its position and axis are relative to component 1. Each component's parameters keep their names with the component's number appended (`x1`, `r1`, `x2`, `s2`, `alpha2`, …); the blur `sigma` is shared. Shape parameters are absolute (a ring's radius and a cap's scale in nm): to make two of them equal, tie them (`params={"r2": ParamToFit(Tied("r1"))}` in `run_locmofit`). `defaults` overrides the generated default starting values.

    Component k explains a share wₖ of the signal localizations, with its points averaged on their own. The shares are softmax(0, `omega2`, …, `omegaK`), so `omega2`, …, `omegaK` (log wₖ/w₁) are fitted, with the background's `omega0` among the fractions of the mixture (LocMoFit.jl's derivations/derivation.qmd, @eq-softmax); they start at 0 (equal shares), and holding the `fractions` (`Fixed()`) fixes them.
    """

    def __init__(
        self,
        *components: SimpleModel,
        defaults: Mapping[str, float] | None = None,
    ):
        self.components = components
        self.defaults = dict(defaults or {})

    def ncomponents(self) -> int:
        return len(self.components)

    def default_parameters(self) -> dict[str, float]:
        theta: dict[str, float] = {}
        for k, c in enumerate(self.components, start=1):
            d = c.default_parameters()
            theta.update({f"{n}{k}": d[n] for n in c.local_names()})
        theta["sigma"] = 1.0
        theta.update(
            {f"omega{k}": 0.0 for k in range(2, len(self.components) + 1)}
        )
        return apply_init(theta, self.defaults)

    def orientation_names(self):
        return tuple(
            (f"theta{k}", f"phi{k}")
            for k in range(1, len(self.components) + 1)
        )

    def component_parameters(self, theta, k: int) -> dict:
        """The parameters of component k (from 1), under the component's own names."""
        c = self.components[k - 1]
        return {n: theta[f"{n}{k}"] for n in c.local_names()}

    def unit_model(self, theta, starts, *, spacing, dtype):
        return tuple(
            (
                c.unit_points(
                    self.component_parameters(theta, k),
                    spacing=spacing,
                    dtype=dtype,
                ),
                rotmat(starts[k - 1][0], starts[k - 1][1], dtype=dtype),
            )
            for k, c in enumerate(self.components, start=1)
        )

    def model_points(self, unit_model, theta):
        # every component k > 1 in component 1's frame; then all of them placed by component 1's pose
        parts = []
        for k, c in enumerate(self.components, start=1):
            pk = self.component_parameters(theta, k)
            points, chart = unit_model[k - 1]
            X = c.local_points(points, pk)
            R = chart_rotation(chart, pk["vx"], pk["vy"])
            if k == 1:
                R1, p1 = R, pk
            else:
                X = place(X, R, pk["x"], pk["y"], pk["z"])
            parts.append(X)
        return place(
            jnp.concatenate(parts, axis=0), R1, p1["x"], p1["y"], p1["z"]
        )

    def log_weights(self, theta) -> tuple:
        # log w₁ = 0 and log wₖ/w₁ = omegak, before normalization
        return (
            jnp.zeros_like(theta["sigma"]),
            *(theta[f"omega{k}"] for k in range(2, len(self.components) + 1)),
        )

    def quantities(self, background: bool) -> dict:
        """Every component's quantities, with its number appended to their names, then the shared ones."""
        out = {}
        for k, c in enumerate(self.components, start=1):
            out.update(c.component_quantities(str(k), k))
        return {**out, **common_quantities(self, background)}

    def summary(self, unit_model, theta) -> list[dict]:
        p1 = self.component_parameters(theta, 1)
        R1 = np.asarray(chart_rotation(unit_model[0][1], p1["vx"], p1["vy"]))
        t1 = np.array([p1["x"], p1["y"], p1["z"]], dtype=float)
        out = []
        for k, c in enumerate(self.components, start=1):
            pk = self.component_parameters(theta, k)
            if k == 1:
                Rk, tk = R1, t1
            else:
                Rk = R1 @ np.asarray(
                    chart_rotation(unit_model[k - 1][1], pk["vx"], pk["vy"])
                )
                tk = t1 + R1 @ np.array(
                    [pk["x"], pk["y"], pk["z"]], dtype=float
                )
            out.append({**pose_summary(tk, Rk[:, 2]), **c.shape_summary(pk)})
        return out


def DoubleRing() -> Composite:  # noqa: N802 (the name of the model, as in LocMoFit.jl)
    """Two rings, the second one placed in the first one's frame, starting 50 nm apart along the axis. For the two rings of a nuclear pore, which have the same radius, tie the radii: `params={"r2": ParamToFit(Tied("r1"))}`."""
    return Composite(Ring(), Ring(), defaults={"z2": 50.0})


def SpCapRing() -> Composite:  # noqa: N802 (the name of the model, as in LocMoFit.jl)
    """A ring and a spherical cap placed in the ring's frame, with the cap's apex starting 100 nm along the ring's axis."""
    return Composite(Ring(), SphericalCap(), defaults={"z2": 100.0})


# canonical names (those of LocMoFit.jl) and aliases, looked up case-insensitively
_MODELS = {
    "ring": Ring,
    "sphericalcap": SphericalCap,
    "doublering": DoubleRing,
    "spcapring": SpCapRing,
}
_ALIASES = {"spcap": "sphericalcap", "npc": "doublering"}
_CANONICAL = ("Ring", "SphericalCap", "DoubleRing", "SpCapRing")


def list_models() -> tuple[str, ...]:
    """The canonical model names."""
    return _CANONICAL


def get_model(name: str):
    """The model with the name `name` (case-insensitive, or an alias such as `spcap` or `npc`)."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {', '.join(_CANONICAL)}"
        )
    return _MODELS[key]()
