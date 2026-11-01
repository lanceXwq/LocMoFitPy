"""Speed of LocMoFitPy's fits, as LocMoFit.jl's benchmarks/speed/speed.jl: for every model on its tutorial site (notebooks/data), from the moved start of the tests, in float64 and float32, the loss and its gradient (jitted) and a complete fit, both a whole `run_locmofit` call (which compiles) and the compiled fit alone, on the padded data, as `run_locmofit` runs it; then 20 "sites" per model (random subsets of 60–100% of the tutorial site's localizations, float64) fitted with `run_locmofit` one after another, which shows how often a dataset compiles. Run from the LocMoFitPy folder:
pixi run -e dev python benchmarks/speed/speed.py
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from common import moved, tutorial  # noqa: E402

import locmofitpy2 as lm  # noqa: E402
from locmofitpy2.loss import fit_data  # noqa: E402
from locmofitpy2.pipeline import compiled_fit  # noqa: E402

CASES = (
    ("ring", "Ring", 3),
    ("ring2D", "Ring", 2),
    ("double_ring", "DoubleRing", 3),
    ("spcap", "SphericalCap", 3),
    ("spcap_ring", "SpCapRing", 3),
)


def best(f, n=5):
    f()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f())
        times.append(time.perf_counter() - t)
    return min(times)


print(f"jax {jax.__version__} on {jax.devices()[0].platform}")
print(
    f"{'case':12s} {'dtype':8s} {'N':>5s} {'M':>6s} {'value ms':>9s} {'grad ms':>9s} {'v ns/NM':>8s} {'g ns/NM':>8s} {'call s':>7s} {'fit s':>7s} {'iter':>5s}"
)
for case, model, D in CASES:
    locs, precs, truth, _, _ = tutorial(case, D)
    init, freeze = moved(truth), ()
    if D == 2:
        init["z"], freeze = truth["z"], ("z",)
    for dtype in ("float64", "float32"):
        kw = dict(
            init_params=init, freeze=freeze, dtype=dtype, fov_volume=300.0**D
        )
        t = time.perf_counter()
        r = lm.run_locmofit(model, locs, precs, **kw)
        call = time.perf_counter() - t
        p, u = lm.fit_problem(model, locs, precs, **kw)
        loss = lm.make_loss(p)
        value, grad = jax.jit(loss), jax.jit(jax.grad(loss))
        tv, tg = best(lambda: value(u)), best(lambda: grad(u))
        data, static = fit_data(p, pad=True)
        fit = compiled_fit(static, 200, 1e-4, False)
        tf = best(lambda: fit(u, data)[0].x, n=3)
        N, M = len(locs), sum(len(rg) for rg in p.ranges)
        print(
            f"{case:12s} {dtype:8s} {N:5d} {M:6d} {1e3 * tv:9.3g} {1e3 * tg:9.3g} {1e9 * tv / (N * M):8.2f} {1e9 * tg / (N * M):8.2f} {call:7.3g} {tf:7.3g} {r['iterations']:5d}"
        )

print(
    f"\n{'case':12s} {'sites':>5s} {'classes':>8s} {'total s':>8s} {'per site s':>11s}"
)
rng = np.random.default_rng(1)
for case, model, D in CASES:
    locs, precs, truth, _, _ = tutorial(case, D)
    init, freeze = moved(truth), ()
    if D == 2:
        init["z"], freeze = truth["z"], ("z",)
    sizes = rng.integers(int(0.6 * len(locs)), len(locs) + 1, 20)
    t = time.perf_counter()
    for n in sizes:
        idx = rng.choice(len(locs), n, replace=False)
        lm.run_locmofit(
            model,
            locs[idx],
            precs[idx],
            init_params=init,
            freeze=freeze,
            fov_volume=300.0**D,
        )
    total = time.perf_counter() - t
    classes = len({lm.loss.bucket(int(n)) for n in sizes})
    print(
        f"{case:12s} {len(sizes):5d} {classes:8d} {total:8.3g} {total / len(sizes):11.3g}"
    )
