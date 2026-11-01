# AGENTS.md

Guidance for AI coding agents working in this repository. Workspace-wide rules (data layout, units, **parity with LocMoFit.jl**, working with the owner) are in `../AGENTS.md`; read that first.

## What this is

**LocMoFitPy** is the JAX port of LocMoFit.jl (`../LocMoFit.jl`). It fits a static parametric geometric model to one site's localizations by maximum likelihood and **should give the same results as LocMoFit.jl**. It is meant to be called from MATLAB (SMAP, via `pyenv`; see `run_locmofit.m`) and from Python (SMAPpy). Python is capped at the newest version the **latest MATLAB release** supports through `pyenv` (R2026b → 3.14, so `<3.15`). Raise the cap when a new MATLAB release supports a newer Python. Once the lab moves to SMAPpy, follow SMAPpy's Python version instead. Everything else (JAX) tracks the latest stable releases (Compatibility in `../AGENTS.md`).

Names differ by layer, so do not "fix" any of them without asking:

- folder: `LocMoFitPy`
- GitHub repo: `lanceXwq/LocMoFitPy2`
- distribution: `LocMoFitPy2`
- import: `locmofitpy2`

Stack: `jax` only (since 2026-10-28; Equinox and optax were dropped when the models moved to a flat parameter vector and the optimizer to an own BFGS loop). Environments are managed by **pixi** (`pyproject.toml` → `[tool.pixi.*]`).

## Commands

Run from the repo root; the tests read CSVs by relative path.

```sh
pixi run -e dev python -m pytest tests -k "not cuda12"   # CPU tests, about 1 minute
pixi run -e dev ruff check src tests
pixi run -e dev mypy src
pixi run -e cuda12dev python -m pytest tests/test_fit_cuda12.py   # Linux + CUDA 12 only (skipped without a GPU)
pixi run -e dev python benchmarks/speed/speed.py                 # speed of the loss, its gradient and the fits
```

Use `python -m pytest`, not the `pytest` entry point. Console-script shebangs in `.pixi/envs` break whenever the folder is moved.

## Layout

The package mirrors LocMoFit.jl's design (2026-10-28), so that a change there is ported module by module.

| Path | Contents |
| --- | --- |
| `src/locmofitpy2/__init__.py` | Public API: `run_locmofit`, `fit_problem`, `make_loss`, `log_likelihood`, `bfgs`, the models (`Ring`, `SphericalCap`, `Composite`, `DoubleRing`, `SpCapRing`), `get_model`, `list_models`, `ParameterMap` |
| `src/locmofitpy2/pipeline.py` | `fit_problem` (everything a fit needs: the `Problem` and the starting free parameters), `run_locmofit` (the entry point MATLAB calls), `compiled_fit` (the jitted fit, cached per model structure and padded size class) and the result dict |
| `src/locmofitpy2/parameters.py` | `ParameterMap` (θ = base + S·u: free parameters → all parameters, with freezing and ties, as LocMoFit.jl's), `apply_init`, `freeze_names`, `tie_pairs` |
| `src/locmofitpy2/models.py` | `SimpleModel` (a shape in its own frame, placed by `x, y, z, vx, vy`), `Ring`, `SphericalCap` (the constant-area cap: `s`, `eta`), `Composite` (components in component 1's frame, suffixed names, weights `omega2…`), the presets `DoubleRing`, `SpCapRing`, and the name lookup `get_model` (case-insensitive; aliases `spcap`, `npc`) |
| `src/locmofitpy2/unit_models.py` | `unit_ring`, `unit_sphere` (Fibonacci), `unit_cap` (direction and depth), as NumPy arrays |
| `src/locmofitpy2/transformations.py` | `rotmat(theta, phi)` (the frame of a starting axis), `chart_rotation(R0, vx, vy)`, `place` |
| `src/locmofitpy2/loss.py` | `log_likelihood` (the mixture's log density summed over localizations, one dimension at a time with the differences formed first, optionally masked; its gradient is a closed-form `custom_vjp`, `_ll_bwd`), `loss_value`, `fit_data` (the arrays and static layout of a fit, padded to `bucket` sizes for `run_locmofit`) and `make_loss(problem, pad=False)` (minus the log posterior per localization, plus the penalty) |
| `src/locmofitpy2/bfgs.py` | `bfgs`: BFGS with LineSearches.jl's backtracking, as Optim.jl's that LocMoFit.jl uses, in a jitted `lax.while_loop`; optional per-iteration history |
| `src/locmofitpy2/_profile.py`, `_jaxprofile.py` | Profiling helpers, gitignored (`_*profile.py`); they predate the port |
| `tests/` | `common.py` (fixed inputs, independent references, tutorial loaders), `test_models.py`, `test_loss.py`, `test_fit.py`, `test_fit_cuda12.py`; `data/julia_fits.jl` writes `data/julia_fits.json`, LocMoFit.jl's fits of the tutorial data, the reference of the parity tests |
| `notebooks/` | Owner's fit notebooks, which still use the API from before the port (`c`, `alpha`, `losses`). `notebooks/data/` holds the tutorial data: `ring`, `ring2D`, `double_ring`, `spcap`, `spcap_ring` (`<case>.csv`, `_truth.csv`, `_parameters.csv`) are copies of LocMoFit.jl's `examples/data`, generated there with DynMoSim; the older `simulated_data_*.csv`, `ground_truth*.csv` are the notebooks'. `fit spcap ROI4.ipynb` runs on experiment data in `../data/`. |
| `benchmarks/speed/` | `speed.py` (as LocMoFit.jl's `benchmarks/speed/speed.jl`) and dated results: `plain_jax_2026-10-28.txt` (the port, before any optimization), `optimistix_2026-10-28.txt` (optimistix's BFGS against the own loop) and `compiled_2026-10-29.txt` (after task 2: compiled once per size class, the closed-form gradient, gradients with each trial value) |
| `NOTES.md` | Implementation notes with measurements: the port (why Equinox and optax went, the optimizer as LocMoFit.jl's), optimistix tried and not adopted, the tolerance. Add to it when a performance or numerical choice is made. Not `note.md`, the owner's notes. |
| `JULIA_DIFFERENCES.md` | How LocMoFitPy differs from LocMoFit.jl: by design, and anything still to port. Update it when LocMoFit.jl changes or a difference is resolved. Differences from MATLAB LocMoFit are in `../LocMoFit.jl/MATLAB_DIFFERENCES.md`. |
| `run_locmofit.m` | MATLAB example (`pyenv` OutOfProcess). It still uses the API from before the port and hard-codes the env path from before the folder moved (`/Users/lancexwq/Ries Lab/LocMoFitPy2/...`). |
| `benchmarks/logprod_vs_sumlog.py`, `profiles/`, `prof.pstats`, `note.md` | Owner's experiments and notes (untracked). Ignore. |

## Models and parameters

As in LocMoFit.jl (its `AGENTS.md`, Adding or changing a model): a model only describes a shape; its parameters live in a flat vector, mapped by `ParameterMap` to a dict `theta` of scalars. A simple model subclasses `SimpleModel` and defines `shape_parameters()` (names and defaults), `unit_points(shape, spacing=, dtype=)` (NumPy, built **once from the starting values**: the point count, `ceil(circumference/spacing)` or `ceil(area/spacing²)`, is fixed for the fit), `local_points(unit, shape)` (JAX, the (M, 3) points in its own frame), `shape_penalty` and `shape_summary`, and gets the pose and blur (`x, y, z, shape…, vx, vy, sigma`). The starting axis comes from `init_params` as `theta`, `phi` and centers the tangent chart; `theta`, `phi` are not parameters (freezing them raises), and the orientation is frozen through `vx`, `vy`. A `Composite` and the presets work as in LocMoFit.jl, with `tie={"r2": "r1"}` for equal parameters. Add a new model to `_MODELS` in `models.py` under LocMoFit.jl's name.

**Mirror every model, default, penalty, or loss change in LocMoFit.jl** (see the parity section in `../AGENTS.md`), or say explicitly that you did not.

## Conventions

- Style (root `AGENTS.md`, Style guide): PEP 8, formatted with `pixi run -e dev ruff format src tests` at 79 characters (`[tool.ruff] line-length = 79` in `pyproject.toml`; ruff skips the gitignored `_*profile.py`) and checked with `pixi run -e dev ruff check --select E,W,I --ignore E501,W505 src tests`. Comments and docstrings have one line per paragraph, list item or signature, with no hard line breaks, so they may exceed 79 characters; commented-out code keeps its lines. The math names (`X`, `R`, `D`, `M`, …) are not renamed for now (the owner's decision: after parity), so the naming rules (`N`) are not part of the check; `DoubleRing`, `SpCapRing` keep LocMoFit.jl's names.
- **Data are `(N, D)` (rows are localizations)**, the transpose of LocMoFit.jl. Precisions are per-axis standard deviations of the same shape, in nm. For 2D input, the model points are truncated to their first 2 columns.
- Parameter names are ASCII (`theta`, `phi`, `alpha`, `sigma`), as in LocMoFit.jl since 2026-10-28 (SciML style: no Unicode in public names).
- `dtype` defaults to `"float64"` (since 2026-10-29, as in LocMoFit.jl; `"float32"` is an option). float64 calls `jax.config.update("jax_enable_x64", True)`, which is **process-global and sticky**: it affects every later JAX call in the same Python process, including a long-lived MATLAB `pyenv` session, so with the default every fit switches it on.
- Everything under `bfgs` is jitted. `run_locmofit` compiles once per model structure (`loss.model_key`), parameter names, padded sizes (`loss.bucket`) and the static options (`max_iter`, `tol`, `history`, background, prior or none), and reuses the compiled fit for every site that matches; the data are arguments, never constants baked into the function. The loss must be pure and traceable: no Python branching on traced values, no `print` (use `jax.debug.print`). If you change `log_likelihood`, change `_ll_bwd` with it: the gradient tests (central differences) and the padding test check them together.
- The value `run_locmofit` returns must stay MATLAB-convertible: a `dict` of NumPy arrays, Python scalars, lists and dicts (`parameters`, `signal_fraction`, `weights`, `components`, `loss`, `model_points`, `iterations`, `retcode`, `history`). Never return JAX arrays.
- Ask before adding dependencies (they go in `pyproject.toml`, and `pixi.lock` changes with them).

## Testing

- `test_models.py`, `test_loss.py` and `test_fit.py` port LocMoFit.jl's suite: the orientation chart against Rodrigues' formula; the model points against their geometry; composites, penalties, freezing and ties; the loss against an independent loop implementation of the posterior (every model, 3D and 2D, with and without background, three priors); float32 at 40 µm against the rounding bound; the change of length unit; JAX's gradient against central differences; `run_locmofit` (frozen values, history, no background, ties); fits of the tutorial data from the moved start, within LocMoFit.jl's tolerances (from its accuracy benchmark); and that the padded loss and gradient `run_locmofit` uses equal the exact ones.
- **Parity:** each tutorial fit (float64) must equal LocMoFit.jl's (`tests/data/julia_fits.json`): the same iterations, the loss to 1e-10 and the parameters to 1e-6. Both use the same BFGS and line search, so they take the same path; on 2026-10-28 they agreed to 12 digits. Regenerate the reference with `julia --project=../LocMoFit.jl tests/data/julia_fits.jl` when LocMoFit.jl's fit changes.
- In float32 the paths part from LocMoFit.jl's, and in both packages float32 fits can stop early when the loss stops changing (`NOTES.md`; accepted, float64 is the default), so there is no float32 fit test.
- Fixed inputs, no RNG (`common.pseudo`); x64 is enabled for the tests (`common.py`).
