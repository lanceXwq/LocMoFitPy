"""The likelihood against an independent reference, in float32 far from the origin, under a change of the length unit, and its gradient, as in LocMoFit.jl's test/runtests.jl."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common import (
    CASES,
    FOV,
    SPACING,
    axes_init,
    fractions,
    make_data,
    parts,
    problem,
    pseudo,
    reference_loss,
    shares,
)

import locmofitpy2 as lm


def test_log_likelihood_float32_far_from_origin():
    # far from the origin (40 µm), single precision keeps each localization's log density: rounding the coordinates to float32 moves each difference by at most u, one unit in the last place at 40 µm (0.004 nm), so the log density of each Gaussian errs by at most Σ_d (2|x_d - y_d|·u + u²)/(2v_d) (LocMoFit.jl's derivations/derivation.qmd), and the log of their mean by at most the largest of these; expanding the square would err by about ε·x²/v ≈ 20
    x = np.array(
        [[10 * pseudo(i, j) for i in range(1, 4)] for j in range(1, 11)]
    )
    y = np.array(
        [[10 * pseudo(i, j + 50) for i in range(1, 4)] for j in range(1, 21)]
    )
    s = np.array(
        [
            [0.5 + abs(pseudo(i, j + 99)) for i in range(1, 4)]
            for j in range(1, 11)
        ]
    )
    sigma, shift = 1.5, 40_000.0

    def logp(dtype, shift):
        return np.array(
            [
                float(
                    lm.log_likelihood(
                        jnp.asarray(x[n : n + 1] + shift, dtype),
                        jnp.asarray(s[n : n + 1], dtype),
                        jnp.asarray(y + shift, dtype),
                        (range(0, 20),),
                        jnp.asarray(sigma, dtype),
                        jnp.zeros(1, dtype),
                        0.0,
                        0.0,
                        False,
                    )
                )
                for n in range(10)
            ]
        )

    u = float(np.spacing(np.float32(shift)))
    bound = np.array(
        [
            max(
                sum(
                    (2 * abs(x[n, d] - y[m, d]) * u + u**2)
                    / (2 * (s[n, d] ** 2 + sigma**2))
                    for d in range(3)
                )
                for m in range(20)
            )
            + 1e-4
            for n in range(10)
        ]
    )
    assert np.all(
        np.abs(logp(np.float32, shift) - logp(np.float64, 0.0)) <= bound
    )


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("D", [3, 2])
@pytest.mark.parametrize(
    "setting",
    [
        "background (5, 2)",
        "background (3, 2)",
        "background, no prior",
        "no background",
    ],
)
def test_loss_against_reference(case, D, setting):
    _, model, theta, starts = CASES[case]
    background = setting != "no background"
    prior = {
        "background (5, 2)": (5.0, 2.0),
        "background (3, 2)": (3.0, 2.0),
    }.get(setting)
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(
        model,
        theta,
        starts,
        locs,
        precs,
        background=background,
        signal_prior=prior,
    )
    thp = p.parameter_map.full_parameters(u)
    assert float(model.penalty(thp)) == 0
    f = 1 / (1 + math.exp(-float(thp["logit_prob"]))) if background else None
    ref = reference_loss(
        locs,
        precs,
        parts(p, thp, D),
        shares(model, thp),
        float(thp["sigma"]),
        f=f,
        V=FOV**D,
        prior=prior,
    )
    assert float(lm.make_loss(p)(u)) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("D", [3, 2])
def test_length_unit(case, D):
    # every length divided by L (positions, radii, scales, the blur, the spacing): the posterior's geometry is the same, and the loss (minus the log posterior per localization) drops by D·log L exactly
    L = 2.5
    _, model, theta, starts = CASES[case]
    scaled = {k: v / L if k[0] in "xyzrs" else v for k, v in theta.items()}
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(model, theta, starts, locs, precs)
    pL, uL = lm.fit_problem(
        model,
        locs / L,
        precs / L,
        init_params={**scaled, **fractions(model), **axes_init(model, starts)},
        spacing=SPACING / L,
        dtype="float64",
        fov_volume=(FOV / L) ** D,
    )
    assert float(lm.make_loss(pL)(uL)) == pytest.approx(
        float(lm.make_loss(p)(u)) - D * math.log(L), rel=1e-10
    )


@pytest.mark.parametrize(
    "case, kwargs, D",
    [
        (0, {}, 3),
        (1, {}, 3),
        (2, dict(freeze=("vx2", "vy2"), tie={"r2": "r1"}), 3),
        (3, {}, 3),
        (2, dict(background=False), 3),
        (3, {}, 2),
    ],
)
def test_gradient(case, kwargs, D):
    # JAX's gradient of the loss against central differences, for every model (with the background, the prior and the weights); DoubleRing with its radii tied and its second orientation frozen, and once without a background; SpCapRing also in 2D
    _, model, theta, starts = CASES[case]
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(model, theta, starts, locs, precs, **kwargs)
    loss = lm.make_loss(p)
    g = np.asarray(jax.grad(loss)(u))
    fd = []
    for i in range(len(u)):
        h = 1e-6 * max(1, abs(float(u[i])))
        fd.append(
            (float(loss(u.at[i].add(h))) - float(loss(u.at[i].add(-h))))
            / (2 * h)
        )
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-8 * np.max(np.abs(fd)))


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("D", [3, 2])
def test_padding(case, D):
    # run_locmofit pads the localizations and each component's points to bucket sizes and masks the padding, so that sites of similar sizes share one compilation: the padded loss and its gradient equal the exact ones
    _, model, theta, starts = CASES[case]
    locs, precs = make_data(model, theta, starts, D=D, every=3)
    p, u = problem(model, theta, starts, locs, precs)
    exact, padded = lm.make_loss(p), lm.make_loss(p, pad=True)
    data, static = lm.loss.fit_data(p, pad=True)
    assert len(data.locs) > len(locs) or any(
        e - s > r.stop - r.start for (s, e), r in zip(static.blocks, p.ranges)
    )
    assert float(padded(u)) == pytest.approx(float(exact(u)), rel=1e-12)
    assert np.allclose(
        jax.grad(padded)(u), jax.grad(exact)(u), rtol=1e-10, atol=1e-14
    )
