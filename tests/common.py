"""Shared test inputs and independent references, as in LocMoFit.jl's test/runtests.jl and test/fit.jl."""

import csv
import math
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import locmofitpy as lm  # noqa: E402

DATA = Path(__file__).parent.parent / "notebooks" / "data"
SPACING = 4.0
FOV = 300.0  # side of the field of view (nm)


def pseudo(i, j):
    """Fixed test inputs (no RNG): a deterministic sequence in (-1, 1), as LocMoFit.jl's (Julia's % keeps the sign)."""
    return math.fmod(math.sin(12.9898 * i + 78.233 * j) * 43758.5453, 1)


# Independent geometry (Rodrigues' formula): the unit vector at polar angle θ and azimuth ϕ, the rotation by θ about (-sin ϕ, cos ϕ, 0) that turns z to it, the minimal rotation turning z to a unit vector m, and the pose of an axis with chart coordinates (vx, vy) around a starting axis (θ₀, ϕ₀): that frame, times the minimal rotation to normalize((vx, vy, 1)) in it.
def direction(theta, phi):
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )


def rodrigues(k, a):
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rotation(theta, phi):
    return rodrigues(np.array([-math.sin(phi), math.cos(phi), 0.0]), theta)


def minrot(m):
    a = np.cross([0.0, 0, 1], m)
    if np.linalg.norm(a) < 1e-15:
        return np.eye(3)
    return rodrigues(a / np.linalg.norm(a), math.acos(m[2]))


def pose(start, vx, vy):
    m = np.array([vx, vy, 1.0])
    return rotation(*start) @ minrot(m / np.linalg.norm(m))


# Parameters away from the defaults, and each component's starting axis (θ₀, ϕ₀), for every model.
CASES = (
    (
        "Ring",
        lm.Ring(),
        dict(x=3.0, y=-2.0, z=5.0, r=40.0, vx=0.1, vy=-0.05, sigma=2.0),
        ((0.4, 1.1),),
    ),
    (
        "SphericalCap",
        lm.SphericalCap(),
        dict(
            x=3.0, y=-2.0, z=5.0, s=35.0, eta=0.6, vx=0.1, vy=-0.05, sigma=2.0
        ),
        ((0.4, 1.1),),
    ),
    (
        "DoubleRing",
        lm.DoubleRing(),
        dict(
            x1=3.0,
            y1=-2.0,
            z1=5.0,
            r1=50.0,
            vx1=0.1,
            vy1=-0.05,
            x2=4.0,
            y2=-3.0,
            z2=45.0,
            r2=55.0,
            vx2=0.05,
            vy2=0.08,
            sigma=2.0,
        ),
        ((0.4, 1.1), (0.2, -0.7)),
    ),
    (
        "SpCapRing",
        lm.SpCapRing(),
        dict(
            x1=3.0,
            y1=-2.0,
            z1=5.0,
            r1=40.0,
            vx1=0.1,
            vy1=-0.05,
            x2=4.0,
            y2=-3.0,
            z2=85.0,
            s2=25.0,
            eta2=0.5,
            vx2=0.05,
            vy2=0.08,
            sigma=2.0,
        ),
        ((0.4, 1.1), (0.2, -0.7)),
    ),
)


def fractions(model):
    """The fractions, away from their defaults: signal fraction logistic(1.2) ≈ 0.77, and composites' shares."""
    return (
        dict(logit_prob=1.2, omega2=0.4)
        if isinstance(model, lm.Composite)
        else dict(logit_prob=1.2)
    )


def axes_init(model, starts):
    """The starting axes as run_locmofit takes them: theta, phi, or thetak, phik for component k."""
    if isinstance(model, lm.Composite):
        return {
            n: v
            for k, s in enumerate(starts, start=1)
            for n, v in ((f"theta{k}", s[0]), (f"phi{k}", s[1]))
        }
    return dict(theta=starts[0][0], phi=starts[0][1])


def unit(model, theta, starts):
    return model.unit_model(theta, starts, spacing=SPACING, dtype=np.float64)


def points(model, theta, starts):
    return np.asarray(model.model_points(unit(model, theta, starts), theta))


def make_data(model, theta, starts, D=3, every=7):
    """Localizations near the model points, and their precisions."""
    Y = points(model, theta, starts)[::every]
    locs = np.array(
        [
            [Y[n, d] + 6 * pseudo(d + 1, n + 1) for d in range(D)]
            for n in range(len(Y))
        ]
    )
    precs = np.array(
        [
            [3 + 4 * abs(pseudo(d + 11, n + 1)) for d in range(D)]
            for n in range(len(Y))
        ]
    )
    return locs, precs


def problem(model, theta, starts, locs, precs, background=True, **kwargs):
    f = (
        fractions(model)
        if background
        else {k: v for k, v in fractions(model).items() if k != "logit_prob"}
    )
    return lm.fit_problem(
        model,
        locs,
        precs,
        init_params={**theta, **f, **axes_init(model, starts)},
        spacing=SPACING,
        dtype="float64",
        fov_volume=FOV ** locs.shape[1],
        background=background,
        **kwargs,
    )


def reference_loss(locs, precs, Ys, w, sigma, f=None, V=1.0, prior=None):
    """The posterior of LocMoFit.jl's derivations/derivation.qmd with plain loops: per localization, the mean of the Gaussian densities over each component's points (variances precision² + σ²), the components weighted by w, mixed with a uniform background of density 1/V at signal fraction f; minus the log posterior with the Beta(a, b) prior, per localization."""
    N, D = locs.shape
    total = 0.0
    for n in range(N):
        v = precs[n] ** 2 + sigma**2

        def density(y):
            return math.prod(
                math.exp(-((locs[n, d] - y[d]) ** 2) / (2 * v[d]))
                / math.sqrt(2 * math.pi * v[d])
                for d in range(D)
            )

        ps = sum(
            w[k] * np.mean([density(y) for y in Y]) for k, Y in enumerate(Ys)
        )
        total += math.log(ps if f is None else f * ps + (1 - f) / V)
    logprior = (
        0.0
        if prior is None or f is None
        else (prior[0] - 1) * math.log(f) + (prior[1] - 1) * math.log(1 - f)
    )
    return -(total + logprior) / N


def parts(prob, theta, D):
    """Each component's points (first D columns), from the parameters."""
    Y = np.asarray(prob.model.model_points(prob.unit_model, theta))
    return [Y[r.start : r.stop, :D] for r in prob.ranges]


def shares(model, theta):
    if isinstance(model, lm.Composite):
        omega = np.array(
            [0.0]
            + [
                float(theta[f"omega{k}"])
                for k in range(2, model.ncomponents() + 1)
            ]
        )
    else:
        omega = np.array([0.0])
    return np.exp(omega) / np.exp(omega).sum()


def full(prob, u):
    return prob.parameter_map.full_parameters(u)


def readcsv(path):
    """A CSV as a dict of float arrays ("true"/"false" as 1/0)."""
    with open(path) as io:
        rows = list(csv.reader(io))
    parse = lambda s: 1.0 if s == "true" else 0.0 if s == "false" else float(s)  # noqa: E731
    return {
        name: np.array([parse(r[k]) for r in rows[1:]])
        for k, name in enumerate(rows[0])
    }


def tutorial(case, D):
    """A tutorial site (notebooks/data, LocMoFit.jl's examples/data): localizations, precisions, true parameters, which localizations are signal and their components."""
    d = readcsv(DATA / f"{case}.csv")
    t = readcsv(DATA / f"{case}_truth.csv")
    ax = ("x_nm", "y_nm", "z_nm")[:D]
    er = ("x_err_nm", "y_err_nm", "z_err_nm")[:D]
    locs = np.stack([d[c] for c in ax], axis=1)
    precs = np.stack([d[c] for c in er], axis=1)
    truth = {
        k: float(v[0])
        for k, v in readcsv(DATA / f"{case}_parameters.csv").items()
    }
    return locs, precs, truth, t["is_signal"] > 0.5, t["component"]


def moved(p):
    """The benchmark's moved start: positions +5 nm, sizes ×1.15 (a cap's κ′ ×0.85), starting axes moved by 0.15 rad in theta and phi, blur 2 nm, the default signal fraction and shares."""
    out = {}
    for k, v in p.items():
        if k in ("x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2"):
            v = v + 5
        elif k in ("r", "r1", "r2", "s", "s2"):
            v = 1.15 * v
        elif k in ("eta", "eta2"):
            v = math.atanh(0.85 * math.tanh(v))
        elif k in ("theta", "phi", "theta1", "phi1", "theta2", "phi2"):
            v = v + 0.15
        elif k == "sigma":
            v = 2.0
        elif k == "logit_prob":
            v = math.log(9)
        elif k == "omega2":
            v = 0.0
        out[k] = v
    return out


def julia_reference(case):
    import json

    return json.loads(
        (Path(__file__).parent / "data" / "julia_fits.json").read_text()
    )[case]


def fit_on_device(case, model, D, device):
    """The tutorial fit of `case` from the moved start, in float64, on `device`."""
    locs, precs, truth, _, _ = tutorial(case, D)
    with jax.default_device(device):
        return lm.run_locmofit(
            model,
            locs,
            precs,
            init_params=moved(truth),
            dtype="float64",
            fov_volume=300.0**D,
        )
