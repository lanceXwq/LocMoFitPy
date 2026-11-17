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
    case_params,
    default_priors,
    every_prior,
    make_data,
    names_of,
    parts,
    pose,
    direction,
    problem,
    pseudo,
    reference_log_prior,
    reference_loss,
    all_fractions,
    start_params,
)

import locmofitpy as lm


def kernel_inputs():
    """10 localizations, their precisions and 20 points, for the tests of the likelihood kernel."""
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
    return x, y, s


def test_log_likelihood_float32_far_from_origin():
    # far from the origin (40 µm), single precision keeps each localization's log density: rounding the coordinates to float32 moves each difference by at most u, one unit in the last place at 40 µm (0.004 nm), so the log density of each Gaussian errs by at most Σ_d (2|x_d - y_d|·u + u²)/(2v_d) (LocMoFit.jl's derivations/derivation.qmd), and the log of their mean by at most the largest of these; expanding the square would err by about ε·x²/v ≈ 20
    x, y, s = kernel_inputs()
    sigma, shift = 1.5, 40_000.0

    @jax.jit
    def one(x, s, y, sigma):  # one localization's log density
        return lm.log_likelihood(
            x,
            s,
            y,
            (range(0, 20),),
            sigma,
            jnp.zeros(1, sigma.dtype),
            0.0,
            False,
        )

    def logp(dtype, shift):
        return np.array(
            [
                float(
                    one(
                        jnp.asarray(x[n : n + 1] + shift, dtype),
                        jnp.asarray(s[n : n + 1], dtype),
                        jnp.asarray(y + shift, dtype),
                        jnp.asarray(sigma, dtype),
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
    # the closed-form gradient (_ll_bwd) against central differences, as LocMoFit.jl tests likelihood_pass: with respect to the points, sigma, the log fractions of two components and the log of the background's term, in 3D and 2D (where the points' third coordinate has no effect)
    x, y, s = kernel_inputs()
    ranges = (range(0, 8), range(8, 20))
    sigma, logw = 1.5, np.log([0.24, 0.56])
    b = math.log(0.2) - 3 * math.log(30.0)

    @jax.jit
    def f(y, sigma, logw, b):
        return lm.log_likelihood(
            jnp.asarray(x[:, :D]),
            jnp.asarray(s[:, :D]),
            jnp.asarray(y),
            ranges,
            jnp.asarray(sigma),
            jnp.asarray(logw),
            b,
            True,
        )

    g = jax.grad(f, argnums=(0, 1, 2, 3))(
        jnp.asarray(y), sigma, jnp.asarray(logw), b
    )
    h = 1e-6

    def fd(k, shift):
        args = [y, sigma, logw, b]
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
    assert close(g[3], fd(3, h))


def prior_variants(model, background):
    """The priors of each variant of the reference test that applies to the model: the defaults, the fractions' default replaced (by a symmetric Dirichlet, by one with an alpha for each fraction, and by a flat density; the fractions need a background or a composite), flat sizes, and every other quantity with a prior."""
    out = {"default": {}}
    n = background + model.ncomponents()
    if background or isinstance(model, lm.Composite):
        out["fractions Dirichlet(2.5)"] = {"fractions": lm.Dirichlet(2.5)}
        out["fractions Dirichlet per fraction"] = {
            "fractions": lm.Dirichlet(*(1.5 + np.arange(1, n + 1)))
        }
        out["fractions flat"] = {"fractions": lm.Flat()}
    out["flat sizes"] = {k: lm.Flat() for k in names_of(model, ("r", "s"))}
    out["every prior"] = every_prior(model)
    return out


VARIANTS = [
    (case, D, background, name)
    for case, (_, model, _, _) in enumerate(CASES)
    for D in (3, 2)
    for background in (True, False)
    for name in (prior_variants(model, background) if D == 3 else ("default",))
]


@pytest.mark.parametrize("case, D, background, variant", VARIANTS)
def test_loss_against_reference(case, D, background, variant):
    # the posterior with its default priors, then with each default replaced (the fractions', and flat sizes) and with every other quantity given a prior, as LocMoFit.jl's test; the log prior does not depend on D, so in 2D only the defaults
    _, model, theta, starts = CASES[case]
    priors = prior_variants(model, background)[variant]
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(
        model, theta, starts, locs, precs, background=background, priors=priors
    )
    thp = p.parameter_map.full_parameters(u)
    used = {**default_priors(model, background), **priors}
    f = all_fractions(model, thp, background)
    ref = reference_loss(
        locs,
        precs,
        parts(p, thp, D),
        f[1:] if background else f,
        float(thp["sigma"]),
        f0=f[0] if background else None,
        V=FOV**D,
        logprior=reference_log_prior(model, thp, starts, used),
    )
    assert float(lm.make_loss(p)(u)) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("alpha", [5.0, 2.0, 1.0])
def test_fraction_prior_optimum(alpha):
    # two rings 2 µm apart along their axis, n1 and n2 localizations exactly on them, no background: every localization comes from its own ring (q = 0 or 1 in LocMoFit.jl's derivations/derivation.qmd, @eq-shares), so the share that maximizes the posterior is (n2 + alpha - 1)/(n + 2(alpha - 1)) (@eq-share-map), n2/n for the flat Dirichlet(1), whose term is 0 (Dirichlet(1) rather than Flat() reuses the compiled fit); the fits also check what a fit without a background returns (as LocMoFit.jl's test)
    n1, n2, r = 30, 10, 50.0

    def ring(n, z):
        t = 2 * math.pi * (np.arange(1, n + 1) - 0.5) / n
        return np.column_stack([r * np.cos(t), r * np.sin(t), np.full(n, z)])

    locs = np.vstack([ring(n1, 0.0), ring(n2, 2000.0)])
    precs = np.full(locs.shape, 5.0)
    theta = dict(
        x1=0.0,
        y1=0.0,
        z1=0.0,
        r1=r,
        theta1=0.0,
        phi1=0.0,
        x2=0.0,
        y2=0.0,
        z2=2000.0,
        r2=r,
        theta2=0.0,
        phi2=0.0,
        sigma=1.0,
    )
    held = start_params(theta, fixed=tuple(start_params(theta)))
    fit = lm.run_locmofit(
        "DoubleRing",
        locs,
        precs,
        params={**held, "fractions": lm.ParamToFit(lm.Dirichlet(alpha))},
        background=False,
        tol=1e-10,
    )
    assert fit["weights"][1] == pytest.approx(
        (n2 + alpha - 1) / (n1 + n2 + 2 * (alpha - 1)), rel=1e-6
    )
    assert (
        "omega0" not in fit["parameters"]
        and fit["signal_fraction"] is None
        and math.isfinite(fit["loss"])
    )


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("D", [3, 2])
def test_length_unit(case, D):
    # every length divided by L (positions, radii, scales, the blur, the spacing, and the log-normal priors' medians): the posterior's geometry is the same, and the loss (minus the log posterior per localization) drops by (D + J/N)·log L exactly, with J log-normal priors (LocMoFit.jl's derivations/derivation.qmd, @eq-units-priors)
    L = 2.5
    _, model, theta, starts = CASES[case]
    scaled = {k: v / L if k[0] in "xyzrs" else v for k, v in theta.items()}
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(model, theta, starts, locs, precs)
    sizes = names_of(model, ("r", "s"))
    pL, uL = problem(
        model,
        scaled,
        starts,
        locs / L,
        precs / L,
        priors={k: lm.LogNormal(math.log(50 / L) + 1, 1) for k in sizes},
        spacing=SPACING / L,
        fov=FOV / L,
    )
    N = len(locs)
    assert float(lm.make_loss(pL)(uL)) == pytest.approx(
        float(lm.make_loss(p)(u)) - (D + len(sizes) / N) * math.log(L),
        rel=1e-10,
    )


@pytest.mark.parametrize(
    "case, kwargs, D",
    [
        (0, {}, 3),
        (
            2,
            dict(
                fixed=("axis2",),
                params={"r2": lm.ParamToFit(lm.Tied("r1"))},
            ),
            3,
        ),
        (2, dict(background=False), 3),
        (3, {}, 2),
        (1, dict(priors=every_prior(CASES[1][1])), 3),
        (3, dict(priors=every_prior(CASES[3][1])), 3),
    ],
)
def test_gradient(case, kwargs, D):
    # JAX's gradient of the loss against central differences, for every model with the background and the weights: Ring with the default priors; SphericalCap and SpCapRing with the default priors and every other kind of prior; DoubleRing with its radii tied and its second axis held, and once without a background; SpCapRing also in 2D (as LocMoFit.jl's test)
    _, model, theta, starts = CASES[case]
    locs, precs = make_data(model, theta, starts, D=D)
    p, u = problem(model, theta, starts, locs, precs, **kwargs)
    loss = jax.jit(lm.make_loss(p))
    g = np.asarray(jax.grad(loss)(u))
    fd = []
    for i in range(len(u)):
        h = 1e-6 * max(1, abs(float(u[i])))
        fd.append(
            (float(loss(u.at[i].add(h))) - float(loss(u.at[i].add(-h))))
            / (2 * h)
        )
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-8 * np.max(np.abs(fd)))


@pytest.mark.parametrize("D", [3, 2])
def test_padding(D):
    # run_locmofit pads the localizations and each component's points to bucket sizes and masks the padding, so that sites of similar sizes share one compilation: the padded loss and its gradient equal the exact ones; on SpCapRing, whose ring and cap give both kinds of component block
    _, model, theta, starts = CASES[3]
    locs, precs = make_data(model, theta, starts, D=D, every=3)
    p, u = problem(model, theta, starts, locs, precs)
    exact = jax.jit(lm.make_loss(p))
    padded = jax.jit(lm.make_loss(p, pad=True))
    data, static = lm.loss.fit_data(p, pad=True)
    assert len(data.locs) > len(locs) or any(
        e - s > r.stop - r.start for (s, e), r in zip(static.blocks, p.ranges)
    )
    assert float(padded(u)) == pytest.approx(float(exact(u)), rel=1e-12)
    assert np.allclose(
        jax.grad(padded)(u), jax.grad(exact)(u), rtol=1e-10, atol=1e-14
    )


def test_priors():
    # where the likelihood does not depend on a quantity, the fit ends at its prior's mode (LocMoFit.jl's derivations/derivation.qmd, Priors on every parameter): the height of a ring in 2D under a normal density; and the cap of a SpCapRing whose share is held at 1e-20, fitted to a ring's localizations with a background, so that the cap's density is negligible next to the background's: its height z2 at the normal's mean, its scale at the log-normal's mode exp(mu - sigma²), its κ′ at the normal's mean, and its axis along ±d under a Bingham density that holds it near the plane perpendicular to ν = (θ, ϕ) = (0.5 + π/2, 1) (κ = −5) and pulls it toward d = (0.5, 1), perpendicular to ν, within that plane (κ = 2): d is the eigenvector of A with the largest eigenvalue (as LocMoFit.jl's test)
    _, model, theta, starts = CASES[0]
    locs, precs = make_data(model, theta, starts, D=2)
    held = tuple(k for k in case_params(model, theta, starts) if k != "z")
    fit = lm.run_locmofit(
        model,
        locs,
        precs,
        params=case_params(
            model, theta, starts, fixed=held, priors={"z": lm.Normal(30, 10)}
        ),
        spacing=SPACING,
        tol=1e-10,
    )
    assert fit["parameters"]["z"] == pytest.approx(30, rel=1e-8)

    ring = dict(x1=3.0, y1=-2.0, z1=5.0, r1=40.0, theta1=0.4, phi1=1.1)
    locs, precs = make_data(
        lm.Ring(),
        dict(x=3.0, y=-2.0, z=5.0, r=40.0, vx=0.0, vy=0.0, sigma=2.0),
        ((0.4, 1.1),),
    )
    start = {
        **ring,
        **dict(
            x2=0.0,
            y2=0.0,
            z2=85.0,
            s2=30.0,
            alpha2=math.pi / 3,
            theta2=0.2,
            phi2=-0.7,
        ),
        "sigma": 2.0,
    }
    free = ("z2", "s2", "kappa2", "axis2")
    priors = dict(
        z2=lm.Normal(70, 5),
        s2=lm.LogNormal(math.log(40), 0.5),
        kappa2=lm.Normal(0.3, 0.2),
        axis2=lm.Bingham(
            lm.Watson(0.5 + math.pi / 2, 1.0, -5.0), lm.Watson(0.5, 1.0, 2.0)
        ),
    )
    params = {
        **start_params(
            start,
            fixed=tuple(k for k in start_params(start) if k not in free),
            priors=priors,
        ),
        "fractions": lm.ParamToFit((0.2, 0.8, 1e-20), lm.Fixed()),
    }
    fit = lm.run_locmofit(
        "SpCapRing", locs, precs, params=params, spacing=SPACING, tol=1e-10
    )
    th = fit["parameters"]
    assert th["z2"] == pytest.approx(70, rel=1e-8)
    assert abs(th["s2"]) == pytest.approx(40 * math.exp(-(0.5**2)), rel=1e-8)
    assert math.copysign(1, th["s2"]) * math.sin(
        th["alpha2"] / 2
    ) == pytest.approx(0.3, rel=1e-8)
    axis = pose((0.2, -0.7), th["vx2"], th["vy2"])[:, 2]
    assert abs(axis @ direction(0.5, 1.0)) == pytest.approx(1, rel=1e-10)


@pytest.mark.parametrize(
    "model, params, background",
    [
        (
            "Ring",
            {"r2": lm.ParamToFit(1.0)},
            True,
        ),  # not a quantity of the model
        ("Ring", {"r": 50.0}, True),  # not a ParamToFit
        (
            "Ring",
            {"x": lm.ParamToFit(lm.LogNormal(0, 1))},
            True,
        ),  # a log-normal fits sizes only
        (
            "Ring",
            {"r": lm.ParamToFit(lm.Watson(0, 0, 1))},
            True,
        ),  # a Watson density fits axes only
        (
            "Ring",
            {"r": lm.ParamToFit(lm.Bingham(lm.Watson(0, 0, 1)))},
            True,
        ),  # and so does a Bingham density
        ("Ring", {"axis": lm.ParamToFit(lm.Normal(0, 1))}, True),
        ("Ring", {"fractions": lm.ParamToFit(lm.Normal(0, 1))}, True),
        (
            "Ring",
            {"fractions": lm.ParamToFit((1.0,))},
            False,
        ),  # no fractions for one component without a background
        (
            "Ring",
            {"fractions": lm.ParamToFit((0.0, 1.0))},
            True,
        ),  # not positive
        (
            "DoubleRing",
            {"fractions": lm.ParamToFit(lm.Dirichlet(2, 5))},
            True,
        ),  # an alpha for each of the three fractions
        ("SphericalCap", {"kappa": lm.ParamToFit(1.0)}, True),
        (
            "Ring",
            {"axis": lm.ParamToFit(0.3)},
            True,
        ),  # an axis is (theta, phi)
        (
            "DoubleRing",
            {"fractions": lm.ParamToFit((0.5, 0.5))},
            True,
        ),  # three fractions with a background
        (
            "DoubleRing",
            {"r2": lm.ParamToFit(lm.Tied("z1"))},
            True,
        ),  # a size tied to a position
        ("DoubleRing", {"r2": lm.ParamToFit(lm.Tied("r3"))}, True),
        ("DoubleRing", {"axis2": lm.ParamToFit(lm.Tied("axis1"))}, True),
    ],
)
def test_params_refused(model, params, background):
    locs = np.array(
        [[10 * pseudo(d, n) for d in range(1, 4)] for n in range(1, 31)]
    )
    with pytest.raises(ValueError):
        lm.fit_problem(
            model,
            locs,
            np.full(locs.shape, 3.0),
            params=params,
            background=background,
        )


@pytest.mark.parametrize(
    "make",
    [
        lambda: lm.Normal(0, 0),
        lambda: lm.LogNormal(0, -1),
        lambda: lm.Dirichlet(0.5),
        lambda: lm.Dirichlet(2, 0.5),
        lambda: lm.Dirichlet(),
        lambda: lm.Bingham(),
    ],
)
def test_prior_parameters_refused(make):
    with pytest.raises(ValueError):
        make()
