"""The cap fit of the tutorial data on a CUDA 12 GPU (`pixi run -e cuda12dev python -m pytest tests/test_fit_cuda12.py`); skipped without one."""

import jax
import pytest
from common import JULIA, tutorial, tutorial_fit

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="no GPU"
)


def test_spcap_fit_cuda12():
    locs, precs, truth, _, _ = tutorial("spcap", 3)
    with jax.default_device(jax.devices("gpu")[0]):
        r = tutorial_fit("SphericalCap", locs, precs, truth)
    assert r["retcode"] == "Success"
    assert r["loss"] == pytest.approx(JULIA["spcap"]["loss"], rel=1e-8)
