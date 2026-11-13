"""What a fresh Python session pays before LocMoFitPy fits fast, as LocMoFit.jl's benchmarks/session/session.jl measures for LocMoFit.jl: the time to import the package, the first fit of every model on its tutorial site (notebooks/data), which compiles it, the median of five fits after it, and the peak memory. No persistent compilation cache is used (the SMAPpy plugin switches one on; JAX_COMPILATION_CACHE_DIR must be unset). The script must be the session's first work, so run it in a new process, from the LocMoFitPy folder:
pixi run -e dev python benchmarks/speed/session.py
"""

import platform
import resource
import statistics
import sys
import time
from pathlib import Path

t0 = time.perf_counter()
import locmofitpy as lm  # noqa: E402

t_load = time.perf_counter() - t0

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from common import tutorial  # noqa: E402

CASES = (
    ("ring", "Ring", 3, "float64"),
    ("spcap", "SphericalCap", 3, "float64"),
    ("double_ring", "DoubleRing", 3, "float64"),
    ("spcap_ring", "SpCapRing", 3, "float64"),
    ("ring2D", "Ring", 2, "float64"),
    ("ring", "Ring", 3, "float32"),
)


def rss():
    """The peak memory in MB (`ru_maxrss` is in bytes on macOS, in kB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def fit_time(model, locs, precs, dtype):
    """The time of one `run_locmofit` call, in s."""
    t = time.perf_counter()
    lm.run_locmofit(model, locs, precs, dtype=dtype)
    return time.perf_counter() - t


print(f"Python {platform.python_version()}, {platform.processor()}")
print(f"import: {t_load:.1f} s, peak memory {rss():.0f} MB")
for case, model, D, dtype in CASES:
    locs, precs = tutorial(case, D)[:2]
    t1 = fit_time(model, locs, precs, dtype)
    t2 = statistics.median(
        fit_time(model, locs, precs, dtype) for _ in range(5)
    )
    label = f"{model} {D}D {dtype}"
    print(
        f"{label:26s} first fit {t1:5.1f} s, then {1e3 * t2:6.1f} ms, "
        f"peak memory {rss():.0f} MB",
        flush=True,
    )
print(f"total since start: {time.perf_counter() - t0:.1f} s")
