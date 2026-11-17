"""Starting values from the data (as LocMoFit.jl's test/runtests.jl, "starting values from the data"; its derivations/derivation.qmd, Starting values from the data): noise-free points with isotropic precisions, so that the expected values follow from the geometry; the two scores against loops over the derivation's formulas; `fit_problem`'s handling of given, frozen and tied values; and the estimate against LocMoFit.jl's on the tutorial data."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from common import direction, minrot, pseudo, rotation, tutorial

import locmofitpy as lm
from locmofitpy import start
from locmofitpy.start import estimate_start
from locmofitpy.unit_models import unit_cap, unit_ring

N_AXIS = direction(0.4, 1.1)
C = np.array([3.0, -2.0, 5.0])
R = rotation(0.4, 1.1)  # turns z to N_AXIS


def axis_of(e, k=""):
    return direction(e[f"theta{k}"], e[f"phi{k}"])


def ring_points():
    return (R @ (40 * unit_ring(np.float64, 120)).T).T + C


def two_rings():
    """Two coaxial rings far apart compared with their radius (r = 35, d = 65 nm), along the coarse candidate axis 30 (29 from 0), and that axis."""
    a = start.hemisphere(start.START_COARSE)[:, 29]
    Ra = minrot(a)
    unit = (Ra @ (35 * unit_ring(np.float64, 60)).T).T
    return np.vstack([unit + (C - 32.5 * a), unit + (C + 32.5 * a)]), a


def scattered():
    """Scattered localizations, two clusters along z, with unequal precisions (40, 3)."""
    locs = np.array(
        [
            [
                10 * pseudo(d, i)
                + (30 * np.sign(pseudo(9, i)) if d == 3 else 0)
                for d in (1, 2, 3)
            ]
            for i in range(1, 41)
        ]
    )
    precs = np.array(
        [[2 + abs(pseudo(d + 20, i)) for d in (1, 2, 3)] for i in range(1, 41)]
    )
    return locs, precs


def test_ring():
    # its center is the mean, its axis the eigenvector with the smallest eigenvalue, its radius every point's distance from the axis
    ring = ring_points()
    e = estimate_start(lm.Ring(), ring, np.full(ring.shape, 2.0))
    assert [e["x"], e["y"], e["z"]] == pytest.approx(C)
    assert e["r"] == pytest.approx(40)
    assert axis_of(e) == pytest.approx(N_AXIS)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_cap(alpha):
    # each point with its mirror image through the axis: the axis is exact and the apex lies on it; flat, the apex is the center and s² is the true s² less half the summed precision variance (here 3·1²), since the mean squared distance from the apex is 2s²
    U = unit_cap(np.float64, 400)
    U = np.vstack([U, np.column_stack([-U[:, :2], U[:, 2]])])
    local = np.asarray(
        lm.SphericalCap().local_points(U, {"s": 30.0, "alpha": alpha})
    )
    cap = (R @ local.T).T + C
    e = estimate_start(lm.SphericalCap(), cap, np.ones(cap.shape))
    assert axis_of(e) == pytest.approx(N_AXIS)  # κ′ ≥ 0, as the true cap's
    apex = np.array([e["x"], e["y"], e["z"]])
    assert np.linalg.norm(np.cross(apex - C, N_AXIS)) < 1e-9
    if alpha == 0:
        assert apex == pytest.approx(C)
        assert e["s"] == pytest.approx(math.sqrt(30**2 - 3 / 2))
        assert abs(e["alpha"]) < 1e-9


def test_scores():
    # the two scores against loops over the derivation's formulas, on the lower group along one axis
    locs, precs = scattered()
    f = start.site_frame(locs, precs)
    m = np.array([0.2, -0.1, 1.0])
    m /= np.linalg.norm(m)
    t = start.AxisTerms(f, m[:, None])
    s = start.sorted_terms(f, t)
    nlow = int(t.nlow[0])
    low = (np.arange(len(f.c)) < nlow)[:, None]
    G = t.order[:nlow, 0]
    w = np.array([f.rel[i] @ m for i in G])
    rho = np.array(
        [math.sqrt(f.rel[i] @ f.rel[i] - (f.rel[i] @ m) ** 2) for i in G]
    )
    along = np.array([np.sum(f.var[i] * m**2) for i in G])
    across = np.array(
        [(np.sum(f.var[i]) - a) / 2 for i, a in zip(G, along, strict=True)]
    )
    cw = f.c[G]

    def mean_w(x):
        return np.sum(cw * x) / np.sum(cw)

    ring = np.sum(
        cw * ((rho - mean_w(rho)) ** 2 + (w - mean_w(w)) ** 2 - across - along)
    )
    assert start.part_scores(lm.Ring(), s, low)[0] == pytest.approx(ring)
    A = np.sqrt(cw)[:, None] * np.column_stack([np.ones(len(G)), rho**2])
    ab = np.linalg.lstsq(A, np.sqrt(cw) * w, rcond=None)[0]
    cap = np.sum(cw * ((w - ab[0] - ab[1] * rho**2) ** 2 - along))
    assert start.part_scores(lm.SphericalCap(), s, low)[0] == pytest.approx(
        cap
    )


def test_two_rings():
    # along a coarse candidate axis, both groups are circles around it, so its score is the lowest and the search returns it exactly, and each ring's start is exact
    two, a = two_rings()
    e = estimate_start(lm.DoubleRing(), two, np.full(two.shape, 2.0))
    assert axis_of(e, 1) == pytest.approx(a)
    assert [e["x1"], e["y1"], e["z1"]] == pytest.approx(C - 32.5 * a)
    assert e["r1"] == pytest.approx(35) and e["r2"] == pytest.approx(35)
    assert e["z2"] == pytest.approx(65) and abs(e["omega2"]) < 1e-12


def test_fit_problem():
    # given values are kept, held ones not given keep their defaults, a tie's target follows its source, and the rest is estimated
    two, a = two_rings()
    precs = np.full(two.shape, 2.0)
    p, u = lm.fit_problem(
        lm.DoubleRing(),
        two,
        precs,
        params={
            "r1": lm.ParamToFit(50.0),
            "z1": lm.ParamToFit(lm.Fixed()),
            "r2": lm.ParamToFit(lm.Tied("r1")),
        },
    )
    theta = p.parameter_map.full_parameters(u)
    assert theta["r1"] == 50 and theta["r2"] == 50 and theta["z1"] == 0
    assert [theta["x1"], theta["y1"]] == pytest.approx((C - 32.5 * a)[:2])
    assert theta["z2"] == pytest.approx(65)
    assert p.unit_model[0][1][:, 2] == pytest.approx(a)
    # a starting axis given is used
    p, _ = lm.fit_problem(
        lm.DoubleRing(),
        two,
        precs,
        params={"axis1": lm.ParamToFit((0.0, 0.0))},
    )
    assert p.unit_model[0][1][:, 2] == pytest.approx([0, 0, 1])
    # in 2D only the position is estimated
    ring = ring_points()[:, :2]
    p, u = lm.fit_problem(lm.Ring(), ring, np.full(ring.shape, 2.0))
    theta = p.parameter_map.full_parameters(u)
    assert [theta["x"], theta["y"]] == pytest.approx(C[:2])
    assert theta["z"] == 0 and theta["r"] == 50
    assert p.unit_model[0][1][:, 2] == pytest.approx([0, 0, 1])
    # a model without an estimate of its shape starts from its defaults, but with component 1 at the localizations' weighted center: here C, about which the two rings are symmetric; the background's fraction starts at 0.1 and the three components' at 0.3 each, so ω₀ = log(0.1/0.3)
    three = lm.Composite(lm.Ring(), lm.Ring(), lm.Ring())
    p, u = lm.fit_problem(three, two, precs)
    theta = p.parameter_map.full_parameters(u)
    position = ("x1", "y1", "z1")
    assert [theta[k] for k in position] == pytest.approx(C)
    assert theta["omega0"] == pytest.approx(math.log(1 / 3))
    defaults = three.default_parameters()
    assert {
        k: v for k, v in theta.items() if k not in (*position, "omega0")
    } == {k: v for k, v in defaults.items() if k not in position}


@pytest.mark.parametrize(
    "model",
    [
        lm.Ring(),
        lm.SphericalCap(),
        lm.DoubleRing(),
        lm.SpCapRing(),
        lm.Composite(lm.SphericalCap(), lm.Ring()),
        lm.Composite(lm.Ring(), lm.Ring(), lm.Ring()),
    ],
)
@pytest.mark.parametrize("D", [3, 2])
def test_start_names(model, D):
    # start_names lists what estimate_start gives, so that fit_problem can skip it when all of them are given
    locs, precs = scattered()
    e = estimate_start(model, locs[:, :D], precs[:, :D])
    assert tuple(e) == start.start_names(model, D)


@pytest.mark.parametrize(
    "case, model, D",
    [
        ("ring", lm.Ring(), 3),
        ("ring2D", lm.Ring(), 2),
        ("spcap", lm.SphericalCap(), 3),
        ("double_ring", lm.DoubleRing(), 3),
        ("spcap_ring", lm.SpCapRing(), 3),
    ],
)
def test_julia_parity(case, model, D):
    # the estimate equals LocMoFit.jl's on the tutorial data (tests/data/julia_starts.json): the same steps, in other floating-point order
    ref = json.loads(
        (Path(__file__).parent / "data" / "julia_starts.json").read_text()
    )[case]
    locs, precs, _, _, _ = tutorial(case, D)
    e = estimate_start(model, locs, precs)
    assert tuple(e) == tuple(ref)
    for k, v in ref.items():
        assert e[k] == pytest.approx(v, rel=1e-9, abs=1e-9)
