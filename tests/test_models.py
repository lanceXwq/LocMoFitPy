"""Geometry, composites, penalties, freezing and ties, as in LocMoFit.jl's test/runtests.jl."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from common import CASES, SPACING, direction, make_data, pose, problem, unit

import locmofitpy as lm
from locmofitpy.transformations import chart_rotation, rotmat


def test_orientation_chart():
    # at its center the chart gives the starting axis; elsewhere the pose of the independent construction, also for a starting axis pointing down
    for start in ((0.4, 1.1), (0.0, 0.0), (math.pi, 0.3), (2.0, -2.5)):
        for vx, vy in ((0.0, 0.0), (0.3, -0.2), (-1.5, 2.0)):
            R = np.asarray(
                chart_rotation(
                    rotmat(*start), jnp.asarray(vx), jnp.asarray(vy)
                )
            )
            assert np.allclose(R, pose(start, vx, vy))
            assert np.allclose(R.T @ R, np.eye(3))
    R = np.asarray(
        chart_rotation(rotmat(0.4, 1.1), jnp.asarray(0.0), jnp.asarray(0.0))
    )
    assert np.allclose(R[:, 2], direction(0.4, 1.1))


def test_ring_points():
    _, model, theta, starts = CASES[0]
    X = np.asarray(model.model_points(unit(model, theta, starts), theta))
    c, n = (
        np.array([theta["x"], theta["y"], theta["z"]]),
        pose(starts[0], theta["vx"], theta["vy"])[:, 2],
    )
    assert len(X) == math.ceil(2 * math.pi * theta["r"] / SPACING)
    assert np.allclose(np.linalg.norm(X - c, axis=1), theta["r"])
    assert np.all(np.abs((X - c) @ n) < 1e-9)


@pytest.mark.parametrize("eta", [0.6, -0.4])
def test_cap_points(eta):
    # the constant-area cap: on the sphere of radius s/κ′ through the apex, within the half-angle acos(1 - 2κ′²) of the axis; its points are uniform in depth below the apex, which on a sphere (the area of a zone is 2πR times its height) means uniform in area; bending the other way for κ′ < 0
    _, model, theta, starts = CASES[1]
    theta = {**theta, "eta": eta}
    X = np.asarray(model.model_points(unit(model, theta, starts), theta))
    kappa, apex, n = (
        math.tanh(eta),
        np.array([theta["x"], theta["y"], theta["z"]]),
        pose(starts[0], theta["vx"], theta["vy"])[:, 2],
    )
    R = theta["s"] / kappa
    center = apex - R * n
    assert len(X) == math.ceil(4 * math.pi * theta["s"] ** 2 / SPACING**2)
    assert np.allclose(np.linalg.norm(X - center, axis=1), abs(R))
    cosine = (X - center) @ n / R
    assert np.all((1 - 2 * kappa**2 - 1e-9 <= cosine) & (cosine <= 1 + 1e-9))
    depth = (
        (apex - X) @ n * np.sign(kappa) / (2 * theta["s"] * abs(kappa))
    )  # 0 at the apex, 1 at the rim
    for q in (0.25, 0.5, 0.75):
        assert abs(np.mean(depth <= q) - q) < 2 / len(X)


def test_flat_cap():
    # flat at κ′ = 0, with radius 2s
    _, model, theta, starts = CASES[1]
    theta = {**theta, "eta": 0.0}
    X = np.asarray(model.model_points(unit(model, theta, starts), theta))
    apex, n = (
        np.array([theta["x"], theta["y"], theta["z"]]),
        pose(starts[0], theta["vx"], theta["vy"])[:, 2],
    )
    assert np.all(np.abs((X - apex) @ n) < 1e-9)
    assert np.all(np.linalg.norm(X - apex, axis=1) <= 2 * theta["s"] + 1e-9)


def test_composite_points():
    # ring 2 sits in ring 1's frame: centered at (x2, y2, z2), radius r2, its axis posed in that frame
    _, model, theta, starts = CASES[2]
    U = unit(model, theta, starts)
    X = np.asarray(model.model_points(U, theta))
    n1 = len(U[0][0])
    R1 = pose(starts[0], theta["vx1"], theta["vy1"])
    c1 = np.array([theta["x1"], theta["y1"], theta["z1"]])
    c2 = c1 + R1 @ np.array([theta["x2"], theta["y2"], theta["z2"]])
    a1, a2 = R1[:, 2], R1 @ pose(starts[1], theta["vx2"], theta["vy2"])[:, 2]
    assert np.allclose(np.linalg.norm(X[:n1] - c1, axis=1), theta["r1"])
    assert np.all(np.abs((X[:n1] - c1) @ a1) < 1e-9)
    assert np.allclose(np.linalg.norm(X[n1:] - c2, axis=1), theta["r2"])
    assert np.all(np.abs((X[n1:] - c2) @ a2) < 1e-9)

    # the cap of SpCapRing: its apex at (x2, y2, z2) in the ring's frame, on the sphere of radius s2/κ′2
    _, model, theta, starts = CASES[3]
    U = unit(model, theta, starts)
    X = np.asarray(model.model_points(U, theta))
    n1 = len(U[0][0])
    R1 = pose(starts[0], theta["vx1"], theta["vy1"])
    c1 = np.array([theta["x1"], theta["y1"], theta["z1"]])
    a2, rho = (
        R1 @ pose(starts[1], theta["vx2"], theta["vy2"])[:, 2],
        theta["s2"] / math.tanh(theta["eta2"]),
    )
    center = (
        c1 + R1 @ np.array([theta["x2"], theta["y2"], theta["z2"]]) - rho * a2
    )
    assert np.allclose(np.linalg.norm(X[:n1] - c1, axis=1), theta["r1"])
    assert np.allclose(np.linalg.norm(X[n1:] - center, axis=1), rho)
    cosine = (X[n1:] - center) @ a2 / rho
    assert np.all(
        (1 - 2 * math.tanh(theta["eta2"]) ** 2 - 1e-9 <= cosine)
        & (cosine <= 1 + 1e-9)
    )


def test_penalties():
    # soft bounds on sizes only (radius and scale ≥ 1 nm), each a squared hinge; orientations need none
    assert float(
        lm.Ring().penalty({**CASES[0][2], "r": 0.5, "vx": 10.0})
    ) == pytest.approx((1 / 0.5**2 - 1) ** 2)
    assert float(
        lm.SphericalCap().penalty({**CASES[1][2], "s": 0.5, "eta": -5.0})
    ) == pytest.approx((1 / 0.5**2 - 1) ** 2)
    # a composite adds its components' penalties; a second ring smaller than the first is not penalized
    dr = lm.DoubleRing()
    assert float(
        dr.penalty({**CASES[2][2], "r1": 0.5, "r2": 0.25})
    ) == pytest.approx((1 / 0.5**2 - 1) ** 2 + (1 / 0.25**2 - 1) ** 2)
    assert float(dr.penalty({**CASES[2][2], "r2": 40.0})) == 0
    assert float(
        lm.SpCapRing().penalty({**CASES[3][2], "s2": 0.5})
    ) == pytest.approx((1 / 0.5**2 - 1) ** 2)


def test_composites():
    # names: each component's own names with its number appended, one shared blur, and the weights
    c = lm.Composite(lm.SphericalCap(), lm.Ring(), defaults={"z2": 30})
    theta = c.default_parameters()
    assert tuple(theta) == (
        "x1",
        "y1",
        "z1",
        "s1",
        "eta1",
        "vx1",
        "vy1",
        "x2",
        "y2",
        "z2",
        "r2",
        "vx2",
        "vy2",
        "sigma",
        "omega2",
    )
    assert (
        theta["s1"] == lm.SphericalCap().default_parameters()["s"]
        and theta["r2"] == 50.0
        and theta["z2"] == 30.0
        and theta["omega2"] == 0
    )
    assert tuple(
        lm.Composite(lm.Ring(), lm.Ring(), lm.Ring()).default_parameters()
    )[-3:] == ("sigma", "omega2", "omega3")
    # with component 1 at the origin and unrotated, each component's points are its own model's points
    theta = {**theta, "vx2": 0.3, "vy2": -0.1, "x2": 5.0}
    starts = ((0.0, 0.0), (0.5, 1.0))
    U = c.unit_model(theta, starts, spacing=SPACING, dtype=np.float64)
    X = np.asarray(c.model_points(U, theta))
    cap_alone = np.asarray(
        lm.SphericalCap().model_points(
            (U[0],),
            dict(
                x=0.0,
                y=0.0,
                z=0.0,
                s=theta["s1"],
                eta=theta["eta1"],
                vx=0.0,
                vy=0.0,
            ),
        )
    )
    ring_alone = np.asarray(
        lm.Ring().model_points(
            (U[1],), dict(x=5.0, y=0.0, z=30.0, r=theta["r2"], vx=0.3, vy=-0.1)
        )
    )
    assert np.allclose(X, np.vstack([cap_alone, ring_alone]))
    # the world-frame summary of each component
    comps = c.summary(U, theta)
    assert np.allclose(comps[1]["position"], [5.0, 0.0, 30.0]) and np.allclose(
        comps[1]["axis"], pose(starts[1], 0.3, -0.1)[:, 2]
    )
    assert comps[1]["r"] == 50.0
    assert comps[0]["kappa"] == pytest.approx(
        math.tanh(theta["eta1"])
    ) and comps[0]["radius"] == pytest.approx(
        theta["s1"] / math.tanh(theta["eta1"])
    )
    assert comps[0]["area"] == pytest.approx(4 * math.pi * theta["s1"] ** 2)


def test_model_names():
    assert lm.list_models() == (
        "Ring",
        "SphericalCap",
        "DoubleRing",
        "SpCapRing",
    )
    assert isinstance(lm.get_model("spcap"), lm.SphericalCap) and isinstance(
        lm.get_model("NPC"), lm.Composite
    )
    with pytest.raises(ValueError):
        lm.get_model("NuclearPoreComplex")


def test_freezing():
    _, model, theta, starts = CASES[0]
    locs, precs = make_data(model, theta, starts)
    m = lm.ParameterMap(theta, ("z", "sigma"))
    u = m.free_parameters(theta)
    assert m.free == ("x", "y", "r", "vx", "vy")
    assert m.full_parameters(u) == theta
    u[m.free.index("r")] = 45.0
    assert m.full_parameters(u) == {**theta, "r": 45.0}
    pf, uf = problem(model, theta, starts, locs, precs, freeze=("z", "sigma"))
    uf = uf.at[pf.parameter_map.free.index("r")].set(45.0)
    p45, u45 = problem(model, {**theta, "r": 45.0}, starts, locs, precs)
    p0, _ = problem(model, theta, starts, locs, precs)
    assert float(lm.make_loss(pf)(uf)) == pytest.approx(
        float(lm.make_loss(p0)(u45))
    )
    with pytest.raises(ValueError):
        lm.ParameterMap(theta, ("w",))
    with pytest.raises(ValueError):
        lm.ParameterMap(theta, tuple(theta))
    # theta, phi set the starting axis; the orientation is frozen through its chart coordinates
    with pytest.raises(ValueError):
        problem(model, theta, starts, locs, precs, freeze=("theta",))


def test_ties():
    _, _, theta, _ = CASES[2]
    m = lm.ParameterMap(theta, ("sigma",), (("r2", "r1"),))
    u = m.free_parameters(theta)
    assert "r2" not in m.free and "sigma" not in m.free

    assert m.full_parameters(u) == {
        **theta,
        "r2": theta["r1"],
    }  # the target takes the source's value
    u[m.free.index("r1")] = 47.0
    assert m.full_parameters(u)["r2"] == 47.0
    # tied to a frozen parameter: frozen at its value; ties chain
    mf = lm.ParameterMap(theta, ("r1",), (("r2", "r1"),))
    assert mf.full_parameters(mf.free_parameters(theta))["r2"] == theta["r1"]
    mc = lm.ParameterMap(theta, (), (("r2", "r1"), ("z2", "r2")))
    assert mc.full_parameters(mc.free_parameters(theta))["z2"] == theta["r1"]
    for args in (
        ((), (("r2", "w"),)),
        ((), (("r2", "r1"), ("r2", "x1"))),
        (("r2",), (("r2", "r1"),)),
        ((), (("r2", "r1"), ("r1", "r2"))),
    ):
        with pytest.raises(ValueError):
            lm.ParameterMap(theta, *args)
