# Differences from LocMoFit.jl

LocMoFitPy is the JAX port of LocMoFit.jl (`../LocMoFit.jl`) and must give the same fitted parameters on the same data, `init_params`, `spacing` and `dtype`. This file lists where the two differ. Its differences from the original MATLAB LocMoFit are in `../LocMoFit.jl/MATLAB_DIFFERENCES.md`.

Status as of 2026-11-01 (LocMoFitPy was ported to LocMoFit.jl's design on 2026-10-28): the same models (`Ring`, `SphericalCap` as the constant-area cap, `Composite`, `DoubleRing`, `SpCapRing`), parameter names, tangent-chart orientation, freezing and ties, likelihood (background, Beta(5, 2) prior, component weights, exact normalization, σ in quadrature), defaults, penalties and optimizer (BFGS with the same backtracking line search and stopping rule). Every tutorial fit in float64 equals LocMoFit.jl's to about 12 digits, with the same number of iterations (`tests/test_fit.py`, parity). New features are developed in LocMoFit.jl first; each appears in the last section here until it is ported.

## By design

These follow from the language and the callers (see the parity section of `../AGENTS.md`):

- **Data layout:** `(N, D)` here, `D×N` in LocMoFit.jl; model points are `(M, 3)` here and `3×M` there.
- **Model lookup:** names are looked up case-insensitively here, with aliases (`spcap`, `npc`); LocMoFit.jl needs the exact name of the type or preset.
- **Ties:** a dict `{"r2": "r1"}` (or pairs) here, `tie=(:r2 => :r1,)` there.
- **Optimizer implementation:** here a JAX transcription of the algorithm Optim.jl runs for LocMoFit.jl (`BFGS(linesearch=BackTracking())`: the same inverse-Hessian update and resets, LineSearches.jl's interpolation, the same stopping rules), jitted in `lax.while_loop`; there Optim.jl itself with Enzyme gradients. Both compute each trial point's value with its gradient (here in the line search, there `objective`).
- **Return value:** a `dict` of NumPy arrays and Python scalars here, which MATLAB can convert, with `iterations` and `retcode` in place of LocMoFit.jl's Optimization.jl `result`, and `components` as a list of dicts. LocMoFit.jl returns a `NamedTuple` `(parameters, signal_fraction, weights, components, loss, model, unit_model, model_points, result, history)`; `history` is opt-in in both (`history=True`), here as `{"losses", "parameters"}` with one entry per iteration and the start.
- **Float32 sums:** XLA sums the log-likelihood over localizations pairwise, LocMoFit.jl sequentially, so float32 fits take different paths. In both, float32 fits can stop early because the loss stops changing in its last bits (`NOTES.md` here and there).
- **Compilation and padding:** `run_locmofit` here compiles one fit per model structure and size class, padding the localizations and points to `loss.bucket` sizes and masking the padding exactly; LocMoFit.jl compiles once per session and type. The likelihood's gradient is a closed-form custom VJP here and a hand-written Enzyme rule there, from the same formulas.

## LocMoFit.jl changes not yet ported

None (2026-11-01).
