"""`run_locmofit`, fits of the tutorial data against the truth, and parity with LocMoFit.jl, as in LocMoFit.jl's test/runtests.jl and test/fit.jl."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from common import (
    CASES,
    JULIA,
    SPACING,
    axes_init,
    case_params,
    direction,
    make_data,
    start_params,
    tutorial,
    tutorial_fit,
    without_chart,
)

import locmofitpy as lm
from locmofitpy.transformations import chart_rotation


def test_run_locmofit():
    _, model, theta, starts = CASES[0]
    locs, precs = make_data(model, theta, starts, every=3)
    init = {
        **without_chart(theta),
        "x": 0.0,
        "z": 0.0,
        "r": 30.0,
        "sigma": 1.0,
        **axes_init(model, starts),
    }
    kw = dict(
        params=start_params(init, fixed=("sigma",)),
        dtype="float64",
        spacing=SPACING,
    )
    r = lm.run_locmofit("Ring", locs, precs, **kw)
    assert r["history"] is None
    assert r["parameters"]["sigma"] == 1.0  # fixed
    assert 0 < r["signal_fraction"] < 1 and r[
        "signal_fraction"
    ] == pytest.approx(1 / (1 + math.exp(r["parameters"]["omega0"])))
    assert np.allclose(r["weights"], [1.0])
    c = r["components"][0]
    assert np.allclose(
        c["axis"],
        np.asarray(
            chart_rotation(
                lm.transformations.rotmat(*starts[0]),
                jnp.asarray(r["parameters"]["vx"]),
                jnp.asarray(r["parameters"]["vy"]),
            )
        )[:, 2],
    )
    assert np.allclose(c["axis"], direction(c["theta"], c["phi"]))
    p0, u0 = lm.fit_problem(lm.Ring(), locs, precs, **kw)
    assert r["loss"] < float(lm.make_loss(p0)(u0))
    assert np.allclose(
        r["model_points"],
        np.asarray(
            lm.Ring().model_points(
                p0.unit_model,
                {k: jnp.asarray(v) for k, v in r["parameters"].items()},
            )
        ),
    )

    h = lm.run_locmofit("Ring", locs, precs, history=True, **kw)
    assert h["parameters"] == r["parameters"]
    assert len(h["history"]["losses"]) == h["iterations"] + 1 > 1
    assert np.all(h["history"]["parameters"]["sigma"] == 1.0)

    # a tie holds throughout; a composite can be passed as a model, and its shares sum to one
    _, model, theta, starts = CASES[2]
    locs, precs = make_data(model, theta, starts)
    t = lm.run_locmofit(
        lm.Composite(lm.Ring(), lm.Ring()),
        locs,
        precs,
        params={
            **case_params(model, theta, starts),
            "r2": lm.ParamToFit(lm.Tied("r1")),
        },
        dtype="float64",
        spacing=SPACING,
        max_iter=20,
    )
    assert t["parameters"]["r2"] == t["parameters"]["r1"]
    assert sum(t["weights"]) == pytest.approx(1) and len(t["weights"]) == 2


def test_point_params():
    # the number of points comes from point_params (ceil(2πr/spacing) for a ring, ceil(4πs²/spacing²) for a cap); the starting values from params, held or not
    _, model, theta, starts = CASES[0]
    locs, precs = make_data(model, theta, starts, every=3)

    def npoints(p):
        return tuple(len(points) for points, _ in p.unit_model)

    kw = dict(dtype="float64", spacing=3.0)
    p80, u80 = lm.fit_problem(
        lm.Ring(), locs, precs, params={"r": lm.ParamToFit(80.0)}, **kw
    )
    p50, u50 = lm.fit_problem(
        lm.Ring(),
        locs,
        precs,
        params={"r": lm.ParamToFit(80.0)},
        point_params={"r": 50.0},
        **kw,
    )
    assert npoints(p80) == (math.ceil(2 * math.pi * 80 / 3),)
    assert npoints(p50) == (math.ceil(2 * math.pi * 50 / 3),)
    assert np.array_equal(u50, u80)
    assert p50.parameter_map.full_parameters(u50)["r"] == 80
    pf, uf = lm.fit_problem(
        lm.Ring(),
        locs,
        precs,
        params={"r": lm.ParamToFit(80.0, lm.Fixed())},
        point_params={"r": 50.0},
        **kw,
    )
    assert npoints(pf) == (math.ceil(2 * math.pi * 50 / 3),)
    assert pf.parameter_map.full_parameters(uf)["r"] == 80
    pc, _ = lm.fit_problem(
        lm.SphericalCap(), locs, precs, point_params={"s": 30.0}, **kw
    )
    assert npoints(pc) == (math.ceil(4 * math.pi * 30**2 / 3**2),)
    # with a tie, the target's points follow the source's value
    pd, ud = lm.fit_problem(
        lm.DoubleRing(),
        locs,
        precs,
        params={
            "r1": lm.ParamToFit(70.0),
            "r2": lm.ParamToFit(lm.Tied("r1")),
        },
        point_params={"r1": 40.0},
        **kw,
    )
    assert npoints(pd) == (math.ceil(2 * math.pi * 40 / 3),) * 2
    assert pd.parameter_map.full_parameters(ud)["r1"] == 70


# Fits of the tutorial data (one simulated site per model, 15% background, a 300 nm field of view) from the moved start, in float64. Every error must stay within 4× the scatter of one fit that LocMoFit.jl's accuracy benchmark measured in the same setting (LocMoFit.jl/benchmarks/accuracy/fractions_prior_2026-11-17.txt, as LocMoFit.jl's test/fit.jl); an axis's error within its mean plus 4× its scatter, and the ring's share of the cap with a ring within 4× its scatter of the share prior's bias there, +0.0075. And the fit must equal LocMoFit.jl's on the same data, which uses the same algorithm (tests/data/julia_fits.json).


def fit(case, model, D):
    locs, precs, truth, signal, component = tutorial(case, D)
    r = tutorial_fit(model, locs, precs, truth)
    # the true structure in the world frame, as run_locmofit reports a fitted one
    p, u = lm.fit_problem(
        model, locs, precs, params=start_params(truth), dtype="float64"
    )
    t = p.model.summary(
        p.unit_model,
        p.parameter_map.full_parameters(u),
    )
    return r, t, signal, component


def signed_deg(a, b):  # a cap's axis has a sign
    return math.degrees(
        math.acos(
            np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
        )
    )


def axis_deg(a, b):  # a ring's axis has no sign
    angle = signed_deg(a, b)
    return min(angle, 180 - angle)


def assert_parity(case, r):
    ref = JULIA[case]
    assert r["iterations"] == ref["iterations"]
    assert r["loss"] == pytest.approx(ref["loss"], rel=1e-10)
    for k, v in ref["parameters"].items():
        assert r["parameters"][k] == pytest.approx(v, rel=1e-6, abs=1e-8)


def test_fit_ring():
    r, (t,), signal, _ = fit("ring", "Ring", 3)
    c = r["components"][0]
    assert r["retcode"] == "Success"
    assert np.all(
        np.abs(c["position"] - t["position"])
        < 4 * np.array([1.254, 1.055, 1.351])
    )
    assert abs(c["r"] - t["r"]) < 4 * 0.5464
    assert axis_deg(c["axis"], t["axis"]) < 3.053 + 4 * 1.833
    assert abs(r["signal_fraction"] - np.mean(signal)) < 4 * 0.01412
    assert_parity("ring", r)


def test_fit_ring_2d():
    # the axis is known only up to its mirror image through the xy plane
    r, (t,), signal, _ = fit("ring2D", "Ring", 2)
    c = r["components"][0]
    mirror = t["axis"] * np.array([1, 1, -1])
    assert r["retcode"] == "Success"
    assert np.all(
        np.abs(c["position"][:2] - t["position"][:2])
        < 4 * np.array([1.321, 1.135])
    )
    assert abs(c["r"] - t["r"]) < 4 * 0.7496
    assert (
        min(axis_deg(c["axis"], t["axis"]), axis_deg(c["axis"], mirror))
        < 8.677 + 4 * 4.957
    )
    assert abs(r["signal_fraction"] - np.mean(signal)) < 4 * 0.01643
    assert_parity("ring2D", r)


def test_fit_double_ring():
    r, t, signal, component = fit("double_ring", "DoubleRing", 3)
    c = r["components"]
    assert r["retcode"] == "Success"
    assert np.all(
        np.abs(c[0]["position"] - t[0]["position"])
        < 4 * np.array([0.9888, 1.065, 2.58])
    )
    assert np.all(
        np.abs(c[1]["position"] - t[1]["position"])
        < 4 * np.array([1.221, 1.389, 2.471])
    )
    assert (
        abs(c[0]["r"] - t[0]["r"]) < 4 * 1.001
        and abs(c[1]["r"] - t[1]["r"]) < 4 * 0.6953
    )
    assert axis_deg(c[0]["axis"], t[0]["axis"]) < 3.842 + 4 * 2.583
    assert axis_deg(c[1]["axis"], t[1]["axis"]) < 4.688 + 4 * 2.661
    assert abs(r["signal_fraction"] - np.mean(signal)) < 4 * 0.01138
    assert abs(r["weights"][0] - np.mean(component[signal] == 1)) < 4 * 0.03133
    assert_parity("double_ring", r)


def test_fit_spherical_cap():
    r, (t,), signal, _ = fit("spcap", "SphericalCap", 3)
    c = r["components"][0]
    assert r["retcode"] == "Success"
    assert np.all(
        np.abs(c["position"] - t["position"])
        < 4 * np.array([3.006, 3.36, 1.734])
    )
    assert abs(c["curvature"] - t["curvature"]) < 4 * 2.008e-4
    assert abs(c["alpha"] - t["alpha"]) < 4 * 0.0344
    assert signed_deg(c["axis"], t["axis"]) < 2.176 + 4 * 1.428
    assert abs(r["signal_fraction"] - np.mean(signal)) < 4 * 0.01511
    assert_parity("spcap", r)


def test_fit_cap_with_ring():
    # the cap's sphere's center is compared, as in the benchmark
    r, t, signal, component = fit("spcap_ring", "SpCapRing", 3)
    c = r["components"]
    center = lambda x: x["position"] - x["radius"] * x["axis"]  # noqa: E731
    assert r["retcode"] == "Success"
    assert np.all(
        np.abs(c[0]["position"] - t[0]["position"])
        < 4 * np.array([1.278, 1.514, 2.282])
    )
    assert abs(c[0]["r"] - t[0]["r"]) < 4 * 0.8043
    assert axis_deg(c[0]["axis"], t[0]["axis"]) < 3.627 + 4 * 1.736
    assert np.all(
        np.abs(center(c[1]) - center(t[1]))
        < 4 * np.array([2.178, 2.412, 11.03])
    )
    assert (
        abs(c[1]["radius"] - t[1]["radius"]) < 4 * 8.923
        and abs(c[1]["alpha"] - t[1]["alpha"]) < 4 * 0.193
    )
    assert signed_deg(c[1]["axis"], t[1]["axis"]) < 4.136 + 4 * 2.222
    assert abs(r["signal_fraction"] - np.mean(signal)) < 4 * 0.01049
    assert (
        abs(r["weights"][0] - np.mean(component[signal] == 1) - 0.0075)
        < 4 * 0.01226
    )
    assert_parity("spcap_ring", r)


@pytest.mark.parametrize(
    "case, model",
    [
        ("ring", "Ring"),
        ("spcap", "SphericalCap"),
        ("double_ring", "DoubleRing"),
        ("spcap_ring", "SpCapRing"),
    ],
)
def test_fit_from_estimated_start(case, model):
    # with no starting values given, every 3D tutorial fit starts from the data and reaches the optimum the moved-start fit reaches (or a better one), the fit that the tests above check against the truth and against LocMoFit.jl's (tests/data/julia_fits.json): its loss no more than 0.01 per localization above, the threshold LocMoFit.jl's DESIGN.md counts a miss by, while the wrong maxima found on simulated sites lay 0.04–0.7 above (as LocMoFit.jl's test/fit.jl)
    locs, precs, _, _, _ = tutorial(case, 3)
    a = lm.run_locmofit(model, locs, precs, fov_volume=300.0**3)
    assert a["retcode"] == "Success"
    assert a["loss"] < JULIA[case]["loss"] + 0.01
