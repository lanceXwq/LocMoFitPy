"""The map from the fitted vector to all parameters, with fixed values and ties (as LocMoFit.jl's ParameterMap)."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping

import numpy as np


class ParameterMap:
    """θ = base + S·u: the map from the fitted vector `u`, which holds only the free parameters, to all parameters θ.

    `base` holds the values of the fixed parameters (zero for the others), and `S` is a constant 0/1 matrix with one 1 per free or tied parameter. `theta0` is the dict of starting values, `fixed` the names of the parameters to keep at them, and `tie` pairs `(target, source)`: the target takes the source's value throughout (its own starting value is ignored), through a 1 in the source's column of `S`. A target tied to a fixed parameter is fixed at that value; ties may chain.
    """

    def __init__(
        self,
        theta0: Mapping[str, float],
        fixed: Iterable[str] = (),
        tie: Iterable[tuple[str, str]] = (),
    ):
        names = tuple(theta0)
        fixed = tuple(fixed)
        tie = tuple(tie)
        targets = tuple(t for t, _ in tie)
        unknown = tuple(
            k
            for k in (*fixed, *targets, *(s for _, s in tie))
            if k not in names
        )
        if unknown:
            raise ValueError(
                f"unknown parameters {unknown}; the parameters are {names}"
            )
        if len(set(targets)) != len(targets):
            raise ValueError(f"a parameter is tied twice: {tie}")
        both = tuple(k for k in targets if k in fixed)
        if both:
            raise ValueError(f"parameters both fixed and tied: {both}")
        source = dict(tie)

        # follow each tie back to a parameter that is not tied itself
        def root(k: str) -> str:
            seen: list[str] = []
            while k in source:
                if k in seen:
                    raise ValueError(f"the ties form a cycle: {tie}")
                seen.append(k)
                k = source[k]
            return k

        free = tuple(k for k in names if k not in fixed and k not in targets)
        if not free:
            raise ValueError(
                "every parameter is fixed or tied; there is nothing to fit"
            )
        self.names = names
        self.free = free
        roots = [root(k) for k in names]
        self.base = np.array(
            [0.0 if r in free else float(theta0[r]) for r in roots]
        )
        self.S = np.array(
            [[1.0 if r == f else 0.0 for f in free] for r in roots]
        )

    def free_parameters(self, theta: Mapping[str, float]) -> np.ndarray:
        """The free parameters of `theta`, as the vector the optimizer works on."""
        return np.array([float(theta[k]) for k in self.free])

    def full_parameters(self, u) -> dict[str, float]:
        """All parameters, fixed and tied ones included, as a dict of floats (outside a fit; inside one, `loss.parameters`)."""
        theta = self.base + self.S @ np.asarray(u, dtype=float)
        return dict(zip(self.names, map(float, theta), strict=True))


def apply_init(
    default: Mapping[str, float], init: Mapping[str, float]
) -> dict:
    """The defaults with the values of `init` in their place; unknown names are ignored with a warning."""
    unknown = tuple(k for k in init if k not in default)
    if unknown:
        warnings.warn(
            f"Ignoring unknown init keys {unknown}; the parameters are {tuple(default)}",
            stacklevel=3,
        )
    return {k: float(init[k]) if k in init else v for k, v in default.items()}
