"""The map from the fitted vector to all parameters, with freezing and ties (as LocMoFit.jl's ParameterMap)."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np


class ParameterMap:
    """θ = base + S·u: the map from the fitted vector `u`, which holds only the free parameters, to all parameters θ.

    `base` holds the values of the frozen parameters (zero for the others), and `S` is a fixed 0/1 matrix with one 1 per free or tied parameter. `theta0` is the dict of starting values, `freeze` the names of the parameters to keep at them, and `tie` pairs `(target, source)`: the target takes the source's value throughout (its own starting value is ignored), through a 1 in the source's column of `S`. A target tied to a frozen parameter is frozen at that value; ties may chain.
    """

    def __init__(
        self,
        theta0: Mapping[str, float],
        freeze: Iterable[str] = (),
        tie: Iterable[tuple[str, str]] = (),
    ):
        names = tuple(theta0)
        freeze = tuple(freeze)
        tie = tuple(tie)
        targets = tuple(t for t, _ in tie)
        unknown = tuple(
            k
            for k in (*freeze, *targets, *(s for _, s in tie))
            if k not in names
        )
        if unknown:
            raise ValueError(
                f"unknown parameters {unknown}; the parameters are {names}"
            )
        if len(set(targets)) != len(targets):
            raise ValueError(f"a parameter is tied twice: {tie}")
        both = tuple(k for k in targets if k in freeze)
        if both:
            raise ValueError(f"parameters both frozen and tied: {both}")
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

        free = tuple(k for k in names if k not in freeze and k not in targets)
        if not free:
            raise ValueError(
                "every parameter is frozen or tied; there is nothing to fit"
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

    def full_vector(self, u, dtype=None):
        """All parameters, frozen and tied ones included, as a vector in the order of `names`."""
        dtype = u.dtype if dtype is None else dtype
        return jnp.asarray(self.base, dtype) + jnp.asarray(self.S, dtype) @ u

    def full_parameters(self, u, dtype=None) -> dict:
        """All parameters as a dict of scalars."""
        theta = self.full_vector(u, dtype)
        return {k: theta[i] for i, k in enumerate(self.names)}


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


def freeze_names(freeze: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(freeze, str):
        return (freeze,)
    return tuple(str(k) for k in freeze)


def tie_pairs(
    tie: Mapping[str, str] | Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Ties as `(target, source)` pairs, from a dict `{target: source}` or a sequence of pairs."""
    if not tie:
        return ()
    items = tie.items() if isinstance(tie, Mapping) else tie
    return tuple((str(t), str(s)) for t, s in items)
