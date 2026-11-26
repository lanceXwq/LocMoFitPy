# LocMoFitPy

[![CI](https://github.com/lanceXwq/LocMoFitPy/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lanceXwq/LocMoFitPy/actions/workflows/CI.yml?query=branch%3Amain)

> [!WARNING]
> LocMoFitPy is experimental: it is under active development, and its API may change without notice.

## Overview

LocMoFitPy fits a geometric model to the localizations of one site in single-molecule localization microscopy (SMLM) data, in 2D or 3D. The models are a ring, a spherical cap, and composites of them, such as two rings (`DoubleRing`) or a cap above a ring (`SpCapRing`). Each model is a dense set of points; a localization's likelihood is a Gaussian mixture over those points, with each localization's own precision and a fitted blur, mixed with a uniform background. The fit maximizes the posterior, with a prior on every fitted quantity, by BFGS on JAX, compiled once per model and size class and reused for every site. Starting values are estimated from the localizations unless given.

LocMoFitPy is the Python port of [LocMoFit.jl](https://github.com/lanceXwq/LocMoFit.jl) and gives the same results; `JULIA_DIFFERENCES.md` lists how the two differ. It ships a plugin for [SMAPpy](https://github.com/ries-lab/SMAPpy) and can be called from MATLAB.

## Installation

Install it from GitHub with pip, into Python 3.12 to 3.15:

```sh
pip install "LocMoFitPy @ git+https://github.com/lanceXwq/LocMoFitPy"
```

On Linux with an NVIDIA GPU, add JAX's CUDA support with the `cuda12` or `cuda13` extra (`"LocMoFitPy[cuda12] @ git+…"`). To develop it, clone the repository and use [pixi](https://pixi.sh): `pixi run -e dev python -m pytest tests -k "not cuda12"` runs the tests.

To use it in SMAPpy, install it into SMAPpy's Python and run `python -m locmofitpy.smappy_setup` once with that Python; the *LocMoFit* plugin then appears in SMAPpy and keeps LocMoFitPy up to date from GitHub.

## Example

Fit a ring to the simulated site in `notebooks/data`, from the repository's root folder:

```python
import numpy as np
from locmofitpy import run_locmofit

d = np.genfromtxt("notebooks/data/ring.csv", delimiter=",", names=True)
locs = np.column_stack([d["x_nm"], d["y_nm"], d["z_nm"]])  # (N, 3) positions, nm
precs = np.column_stack([d["x_err_nm"], d["y_err_nm"], d["z_err_nm"]])  # (N, 3) precisions (standard deviations), nm

fit = run_locmofit("Ring", locs, precs)
fit["components"][0]  # the ring's position, axis and radius
fit["signal_fraction"]  # the share of localizations that are not background
```

The docstring of `run_locmofit` describes its options: the models, starting values, priors, held and tied parameters, 2D data and float32.

## Documentation

- `run_locmofit`'s docstring (`help(run_locmofit)`) — how to call a fit and what it returns.
- `run_locmofit.m` — calling LocMoFitPy from MATLAB.
- `DESIGN.md` — why the code is implemented the way it is, with the measurements behind each choice.
- `JULIA_DIFFERENCES.md` — how LocMoFitPy differs from LocMoFit.jl. The differences from MATLAB LocMoFit are in LocMoFit.jl's `MATLAB_DIFFERENCES.md`.

## License

MIT; see `LICENSE`.
