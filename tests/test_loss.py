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

import locmofitpy as lm


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


@pytest.mark.parametrize("D", [3, 2])
def test_log_likelihood_gradient(D):
    # the closed-form gradient (_ll_bwd) against central differences, as LocMoFit.jl tests likelihood_pass: with respect to the points, sigma, the log weights of two components and the two log fractions of the mixture with the background, in 3D and 2D (where the points' third coordinate has no effect)
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
    ranges = (range(0, 8), range(8, 20))
    sigma, logw = 1.5, np.log([0.3, 0.7])
    a, b = math.log(0.8), math.log(0.2) - 3 * math.log(30.0)

    def f(y, sigma, logw, a, b):
        return lm.log_likelihood(
            jnp.asarray(x[:, :D]),
            jnp.asarray(s[:, :D]),
            jnp.asarray(y),
            ranges,
            jnp.asarray(sigma),
            jnp.asarray(logw),
            a,
            b,
            True,
        )

    g = jax.grad(f, argnums=(0, 1, 2, 3, 4))(
        jnp.asarray(y), sigma, jnp.asarray(logw), a, b
    )
    h = 1e-6

    def fd(k, shift):
        args = [y, sigma, logw, a, b]
        plus, minus = list(args), list(args)
        plus[k], minus[k] = args[k] + shift, args[k] - shift
        return (float(f(*plus)) - float(f(*minus))) / (2 * h)

    def e(shape, index):
        out = np.zeros(shape)
        out[index] = h
        return out

    def close(got, want):
        return np.linalg.norm(np.asarray(got) - want) <= 1e-6 * np.linalg.norm(
            want
        )

    points = np.array(
        [[fd(0, e(y.shape, (m, d))) for d in range(3)] for m in range(20)]
    )
    assert close(g[0], points)
    if D == 2:
        assert np.all(np.asarray(g[0])[:, 2] == 0)
    assert close(g[1], fd(1, h))
    assert close(g[2][0], fd(2, e(2, 0))) and close(g[2][1], fd(2, e(2, 1)))
    assert close(g[3], fd(3, h)) and close(g[4], fd(4, h))


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
@pytest.mark.parametrize("alpha", [5.0, 2.5, None])
def test_loss_against_reference(case, D, setting, alpha):
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
        share_prior=alpha,
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
        share_prior=alpha,
    )
    assert float(lm.make_loss(p)(u)) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("alpha", [5.0, 2.0, None])
def test_share_prior_optimum(alpha):
    # two rings 2 µm apart along their axis, n1 and n2 localizations exactly on them, no background: every localization comes from its own ring (q = 0 or 1 in LocMoFit.jl's derivations/derivation.qmd, @eq-shares), so the share that maximizes the posterior is (n2 + alpha - 1)/(n + 2(alpha - 1)) (@eq-share-map), and n2/n without the prior (as LocMoFit.jl's test)
    n1, n2, r = 30, 10, 50.0

    def ring(n, z):
        t = 2 * math.pi * (np.arange(1, n + 1) - 0.5) / n
        return np.column_stack([r * np.cos(t), r * np.sin(t), np.full(n, z)])

    locs = np.vstack([ring(n1, 0.0), ring(n2, 2000.0)])
    precs = np.full(locs.shape, 5.0)
    model = lm.get_model("DoubleRing")
    theta = {
        **model.default_parameters(),
        "r1": r,
        "r2": r,
        "z2": 2000.0,
        "sigma": 1.0,
    }
    fixed = tuple(k for k in theta if k != "omega2")
    fit = lm.run_locmofit(
        model,
        locs,
        precs,
        init_params=theta,
        freeze=fixed,
        background=False,
        share_prior=alpha,
        tol=1e-10,
    )
    expected = (
        n2 / (n1 + n2)
        if alpha is None
        else (n2 + alpha - 1) / (n1 + n2 + 2 * (alpha - 1))
    )
    assert fit["weights"][1] == pytest.approx(expected, rel=1e-6)


def test_share_prior_below_one():
    locs = np.zeros((20, 3))
    with pytest.raises(ValueError):
        lm.fit_problem("Ring", locs, np.ones_like(locs), share_prior=0.5)


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
