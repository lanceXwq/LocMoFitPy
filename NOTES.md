# Implementation notes

Why LocMoFitPy is implemented the way it is, with the measurements behind each choice. Its features and math are LocMoFit.jl's (`../LocMoFit.jl/NOTES.md`, `../LocMoFit.jl/derivations/derivation.qmd`); how to use and change the code is in `AGENTS.md`.

## The port to LocMoFit.jl's design (2026-10-28)

LocMoFitPy was rebuilt on LocMoFit.jl's design: models that only describe a shape, a flat parameter vector with `ParameterMap` for freezing and ties, composites with component weights, the likelihood with background and prior, the tangent chart and the constant-area cap. Equinox (models as pytrees, freezing by leaf name) and optax (L-BFGS) were no longer used and were dropped; ties in particular do not fit Equinox's partition by name.

**The optimizer is LocMoFit.jl's, transcribed** (`bfgs.py`): Optim.jl's BFGS with LineSearches.jl's `BackTracking` (the same inverse-Hessian update and resets, the same cubic interpolation and step limits, the same stopping rules), in a jitted `lax.while_loop`. optax has no BFGS, and L-BFGS did not converge in LocMoFit.jl on the double ring and the caps. With the same algorithm, every tutorial fit in float64 equals LocMoFit.jl's to about 12 digits, with the same number of iterations (`tests/test_fit.py`), so parity is checked exactly rather than to the optimizer's tolerance.

**optimistix, tried and not adopted** (`benchmarks/speed/optimistix_2026-10-28.txt`). Its BFGS (version 0.1.0, backtracking Armijo search), warm and jitted, on the five tutorial sites. Its built-in termination is Cauchy's (the changes of the parameters and of the loss both below `atol + rtol·|value|`), with no gradient tolerance; a subclass that overrides `terminate` with our rule (largest gradient component ≤ 1e-4) makes the two comparable:

- **Float64, gradient rule:** the same optimum (loss within 1e-7 per localization, parameters within 0.015) in a similar number of steps, and 15–35% faster (ring 6.8 against 10.4 ms, double ring 36 against 53 ms, cap 0.65 against 0.76 s, cap with ring 0.22 against 0.32 s). The steps are not fewer; the evaluations are: optimistix evaluates the loss at each trial point with `jax.linearize` and, when it accepts the point, gets the gradient from that linearization (one backward pass) without evaluating the loss again, while our loop, as Optim.jl, evaluates the loss at the trial point and then the loss and its gradient again once the point is accepted. That is what LocMoFit.jl's `objective` avoids by caching, and our loop can do the same (task 2).
- **Float32:** with the gradient rule optimistix stalls on the ring, the cap and the cap with ring: its line search (halving, no interpolation) finds no decrease the float32 loss resolves, and it runs into the step limit with the largest gradient component at 2e-4 to 4.5e-4; with its Cauchy rule it stops there instead and reports success. Our loop, with LineSearches.jl's interpolating backtracking, reaches the tolerance.
- **Parity:** a different line search takes a different path, so fits would match LocMoFit.jl only to the optimizer's tolerance.

So optimistix would add Equinox and optimistix as dependencies; its Float64 advantage comes from fewer evaluations, which our loop can get too, and it is less reliable in Float32.

## The tolerance

`tol = 1e-4` on the largest gradient component (of the loss per localization) leaves the parameters close to the optimum compared with their statistical error, but not exactly at it: on the cap with a ring, optimistix's tighter fit moved a parameter by 0.53 (float64) and lowered the loss by 2.4e-5 per localization, 0.008 in log-likelihood over the site. The cap's position scatters by 2–11 nm over fits (LocMoFit.jl's accuracy benchmark), so this is far inside it.

## Speed (task 2, 2026-10-29)

Measured with `benchmarks/speed/speed.py` (`plain_jax_2026-10-28.txt` before, `compiled_2026-10-29.txt` after), Apple M1 Pro, CPU.

**Compile once, not per call.** `run_locmofit` used to jit a new function for every call, and a call cost 0.8–1.8 s, of which tracing and compiling took 0.5–1 s and the fit itself 0.01–1 s. Now a compiled fit takes the data as arguments and is cached per model structure and array layout (`pipeline.compiled_fit`). Since JAX compiles once per array shape, the localizations and each component's points are padded to sizes with at most 4 significant bits (`loss.bucket`: …, 112, 120, 128, 144, …; at most 1/8 wasted) and the padding is masked exactly (zero weight; -∞ exponents; the true counts in the normalization); `tests/test_loss.py` checks that the padded loss and gradient equal the exact ones. A second call on the same site then costs the fit: a ring 16 ms instead of 1.0 s, a double ring 65 ms instead of 0.95 s. Twenty sites of 60–100% of a tutorial site's size fall into 5–7 size classes, so a dataset compiles once per model and size class (each about 0.5–0.8 s), not per site.

**The likelihood one dimension at a time, with its gradient in closed form.** Writing the exponents as a sum over the D dimensions of N×M arrays, instead of one N×M×D array, gave the same values to the bit and made a value 1.6–3.5× faster. The gradient is a `jax.custom_vjp` (`loss._ll_bwd`) from the shares of LocMoFit.jl's derivation (as LocMoFit.jl's rule): the backward pass recomputes the exponents instead of storing them, forms the gradient with respect to the points by two matrix products on coordinates centered at the localizations' mean, and keeps the differences for the gradient with respect to the variances (expanding that square would cancel in float32). It equals autodiff's gradient to 3e-16 (float64) and 3e-7 (float32); on the cap a gradient took 7.8 instead of 25.6 ms (float64) and 3.4 instead of 8.2 ms (float32). `jax.checkpoint` did not help.

**The gradient with every trial value.** As optimistix showed, our loop evaluated the loss at the trial point and then the loss and its gradient again once it accepted the point; the line search now evaluates both at each trial point, so an accepted point needs nothing more. Fits took 15–35% less (the ring 9.5 → 6.0 ms, the double ring 46 → 30 ms, the cap 0.33 → 0.28 s in float64), with the same iterations and results (the parity tests pass).

**Result.** A cap fit as `run_locmofit` runs it, compiled and on padded data, takes 0.32 s (float64) and 0.15 s (float32), against LocMoFit.jl's 0.29 and 0.18 s; the cap with ring 0.20 and 0.11 s (LocMoFit.jl 0.12 and 0.07 s); a ring 6.6 ms (LocMoFit.jl 2.8 ms), where the loop's fixed cost per iteration dominates (`compiled_padded_2026-11-01.txt`). Padding costs about 10% here: `compiled_2026-10-29.txt` timed the same loop on unpadded data (the cap 0.29 s in float64).

## How many compilations a dataset costs (2026-11-06)

Fitting many sites, as SMAPpy's per-region steps do, compiles once per setup and size class, never once per site. Measured with JAX 0.11.2 on an Apple M1 Pro, CPU, 3D, float64, through `run_locmofit` and through the SMAPpy plugin on 400 ring sites.

**What decides a compilation.** There are two levels. `pipeline.compiled_fit` (an `lru_cache`) holds one jitted fit per key: the `FitStatic` (the model's structure, all parameter names, each component's padded point blocks, background on or off, prior on or off, padding on), plus `max_iter`, `tol` and `history`. JAX then compiles each jitted fit once per shape and dtype of its arguments: the padded number of localizations, D, the dtype, and the number of free parameters (the length of `u0` and the columns of `S`). Everything else is an argument and never compiles: the data, the starting values that leave the point count alone, the prior's (a, b) and the field-of-view volume. So the number of compilations is the number of setups used times the size classes their sites fall into.

**Size classes.** `loss.bucket` rounds a count up to a number with at most 4 significant bits, in steps of 2^(bit length − 4):

| Count | Step | Classes |
| --- | --- | --- |
| 1–16 | 1 | each count its own |
| 17–32 | 2 | 18, 20, …, 32 |
| 33–64 | 4 | 36, 40, …, 64 |
| 65–128 | 8 | 72, 80, …, 128 |
| 129–256 | 16 | 144, 160, …, 256 |
| 257–512 | 32 | 288, 320, …, 512 |

That gives 8 classes per doubling of N, so sites with n to 2n localizations need at most 8 compilations and sites with 50 to 500 at most about 28, with padding always under 1/8. Each component's points are rounded the same way, and their count is fixed by the starting values (or `point_params`) and `spacing`: a ring has ceil(2πr / spacing) points, 105 at r = 50 nm and spacing 3 nm, class 112, which every starting r from 49.7 to 53.4 nm shares.

**What compiles again**, one ring site of 101 localizations each time, after a first fit of 102 (class 104):

| Change | Compilations of the fit | XLA compile time |
| --- | --- | --- |
| first fit of the process | 1, and about 20 small eager operations | 1.5 s |
| N in a class already compiled | none | 0 |
| N in a new class (90, class 96) | 1 | 0.8 s |
| starting position (`x = 5`) | none | 0 |
| starting r = 60 nm (126 points, class 128) | 1 | 1.5 s |
| `freeze=("r",)` | 1 | 1.1 s |
| `max_iter=300` | 1 | 0.9 s |
| `background=False` | 1 | 0.9 s |
| 2D instead of 3D | 1 | 0.8 s |
| float32 | 1 | 0.8 s |

**Cost per model.** A site in a new size class costs, beyond its fit and with tracing included, about 1.0 s for a ring, 1.2 s for a spherical cap, 1.4 s for a cap with a ring and 1.5 s for a double ring (tutorial sites). Through the SMAPpy plugin, 400 ring sites (93–127 localizations after the layer's filters) fell into 5 classes: 5 compilations of the fit, about 4 s in all, and 6 ms per site for the rest.

**Kept only in the process.** The compiled fits live in the Python process, so every new process (a SMAPpy session, a MATLAB `pyenv` session) compiles again. For real data whose sites span 50 to 500 localizations that is up to about 28 compilations, 30–40 s per session and setup. The SMAPpy plugin keeps them on disk across sessions (the section after next).

## One point count for sites with different starts (`point_params`, 2026-11-12)

**The problem.** The number of model points comes from the starting values (a ring's ceil(2πr / spacing)) and is a shape of the compiled fit. Starting values estimated from each site's localizations, which an automatic start for the SMAPpy plugin would use, give each site its own point count, so the size classes of the points multiply those of N.

**The option.** `point_params` (also in LocMoFit.jl, for parity) replaces the starting values only for generating the points. Sites fitted with the same `point_params` have one point count per component, as sites that all start from the defaults have. LocMoFit.jl does not need it for speed: Julia compiles per argument type, not per array length, and after the first fit of a ring had compiled (43 s, with Enzyme) a fit took 2–4 ms for every point count from 63 to 168.

**Measured** with JAX 0.12.0.1, Apple M1 Pro, CPU, 3D, float64, one fresh process per row, no persistent cache. DynMoSim sites, each with its own true parameters (200 rings: r 30–80 nm, tilt up to 60°; 100 caps: s 30–60 nm, κ′ 0.1–0.9, tilt up to 45°; the center up to 30 nm from the ROI center laterally and 60 nm axially; 15% background), each fitted once. The estimated start comes from moments of the localizations (a scratch test, not in the package):

| | Point counts | Compiled fits | XLA compiling | Session |
| --- | --- | --- | --- | --- |
| Ring, start at the defaults (r = 50 nm) | 1 | 8 | 7.4 s | 13.6 s |
| Ring, start estimated per site | 96 | 60 | 50.6 s | 76.7 s |
| Ring, the same start, `point_params={"r": 50}` | 1 | 8 | 7.3 s | 12.7 s |
| Cap, start at the defaults (s = 50 nm) | 1 | 4 | 4.1 s | 30.5 s |
| Cap, start estimated per site | 98 | 43 | 40.7 s | 73.5 s |
| Cap, the same start, `point_params={"s": 50}` | 1 | 4 | 3.9 s | 22.6 s |

The fits that reached the optimum gave the same values with a point count per site and with one for all sites (position error median 1.86 nm and r error rms 0.84 nm for the rings, s error rms 1.13 nm for the caps), although one count for all puts the points of an 80 nm ring 4.8 nm apart and those of a 30 nm ring 1.8 nm apart.

## Starting values from the data (the port, 2026-11-13)

The port of LocMoFit.jl's `estimate_start` (`start.py`); the choices, and the misses of the optimum it avoids, are in LocMoFit.jl's `NOTES.md` and its derivation. The direction search here scores each set of candidate axes at once, as (N, K) arrays, where LocMoFit.jl loops over the candidates; a Python loop was not tried.

**Parity.** On the 500 sites of LocMoFit.jl's `benchmarks/start/` (generated there with DynMoSim), in 3D and in 2D, every estimate has the same names as LocMoFit.jl's and the same values to 2·10⁻¹⁴ relative, so the search picks the same candidate axis on every site (scratch comparison, 2026-11-13). The tests compare the tutorial sites (`tests/data/julia_starts.json`).

**Cost**, median per site (3D, float64, Apple M1 Pro, CPU): 0.10 ms for a ring, 0.14 ms for a cap, 2.7 ms for a double ring and 4.9 ms for a cap and a ring, against 0.04, 0.15, 1.0 and 2.2 ms in LocMoFit.jl; small next to a fit.

**Point counts.** A start estimated per site gives every site its own point count unless `point_params` is set, and so many more compilations (the section on `point_params`).

## The share prior (the port, 2026-11-14)

The port of LocMoFit.jl's prior on the components' shares (its `NOTES.md`, The share prior, and its derivation, A prior on the shares): (alpha − 1) Σₖ log wₖ added to the log posterior, alpha = 5 by default. alpha is one of the arrays a compiled fit takes (`FitData.share_prior`), not part of its static layout, so a fit with another alpha, or none (passed as alpha = 1, for which the term is exactly 0), reuses the same compilation. The scratch study behind the choice was run with LocMoFitPy, the prior patched into `loss_value`; the parity references (`tests/data/julia_fits.json`, `julia_starts.json`) were written again with LocMoFit.jl's share prior on.

## Priors on every parameter (the port, 2026-11-14 on the branch `priors`; merged into main 2026-11-17)

The port of LocMoFit.jl's prototype on its branch `priors` (its `NOTES.md`, Priors on every parameter; its derivation, Priors on every parameter): `params`, a dict of `ParamToFit(value, prior)` per quantity, replaces `init_params`, `freeze`, `tie`, `signal_prior` and `share_prior`; holding and tying are the priors `Fixed()` and `Tied("r1")`; the size penalties are replaced by a log-normal prior on sizes (`LogNormal(log(50), 1)` at first, `LogNormal(log(50) + 1, 1)` since the same day: below); the other defaults are unchanged.

**Compilations.** Which priors a fit has, each term's `(kind, names, family)`, is part of `FitStatic`, so a fit is compiled for its set of priors; their numbers are arrays in `FitData`, so other numbers reuse the compilation. A dataset fitted with the default priors compiles as before (the same size classes in `benchmarks/speed/speed.py`'s 20 sites per model), while a new family or a prior on another quantity compiles again, like a new model. So "no prior" on the shares is best written `Dirichlet(1)`, whose term is exactly 0, rather than `Flat()`, where it matters (the share-prior test does so).

**Speed** (`benchmarks/speed/speed.py`, twice each on `main` and the branch): the compiled fits took the same time within run-to-run noise for every model in float64, and a whole `run_locmofit` call (compilation included) as well. Some float32 fits took a different number of iterations, since the objective changed (the ring 29 → 43, the cap 41 → 51, the cap with a ring 94 → 84). The jitted gradient of the cap with a ring alone, outside a fit, was 15–20% slower in both runs on the branch (2.5 against 2.1–2.2 ms); it does not show in the fit, and was not looked into.

**Parity.** `tests/data/julia_fits.jl` passes `params` and was run against LocMoFit.jl's branch `priors`: every tutorial fit moved a little (its loss now has the log-normal terms) and still equals LocMoFit.jl's, with the same iterations; `julia_starts.json` did not change.

**The cap's half-angle and the size prior's mode (2026-11-14, as LocMoFit.jl's branch `priors`, its NOTES.md).** The cap is fitted by its half-angle `alpha`, κ′ = sin(`alpha`/2), instead of `eta`, κ′ = tanh(`eta`), which put the whole sphere at `eta` → ∞, where fits got stuck. The default size prior is `LogNormal(log(50) + 1, 1)`, most probable at 50 nm instead of 18 nm. `julia_fits.json` and `julia_starts.json` were regenerated from LocMoFit.jl's branch (the tutorial sizes moved by at most 0.044 nm; the cap's `eta` became `alpha`), and every tutorial fit still equals LocMoFit.jl's. Speed (`benchmarks/speed/speed.py`, twice): the compiled fits took the same time within run-to-run noise; some float32 fits took another path (the cap 51 → 49 iterations, the cap with a ring 84 → 94). The tutorial's parameter files (`notebooks/data/`) still name a cap's curvature `eta` (κ′ = tanh eta), which `tests/common.py` reads alongside `alpha`.

**A Bingham prior on axes (2026-11-16, as LocMoFit.jl's branch `priors`, its NOTES.md).** `Bingham(*terms)` adds the exponents of several Watson densities, so an axis can be held almost exactly in the image plane while its turn stays in a soft range (`Bingham(Watson(0, 0, -1000), Watson(math.pi / 2, -math.pi / 2, 2.6))`: tilt SD about 1.3°, turn about 25°). Watson and Bingham both compile to one quadratic form in the chart's coordinates (family `"quadratic"`, six numbers as data), so changing a κ does not recompile. Tested as LocMoFit.jl tests it; the tutorial fits (`julia_fits.json`) use no axis priors and did not change.

**The background as fraction 0 (2026-11-17, as LocMoFit.jl's branch `priors`, its NOTES.md).** The signal fraction and the shares are one softmax over the background and the components, `(f0, f1, …, fK) = softmax(omega0, 0, omega2, …)`, with one quantity `fractions` and one prior, `Dirichlet(2, 5, …, 5)` with a background and `Dirichlet(5)` without; `Beta`, `logit_prob`, `signal` and `shares` are gone, and the kernel lost its signal argument `a` (`log_likelihood(locs, loc_prec, points, ranges, sigma, logw, b, background)`). A Dirichlet with one alpha per fraction is passed to the fit as its alphas, so a symmetric one and the default share a compilation. `julia_fits.json` was regenerated: the simple models' fits are unchanged (ω₀ = −`logit_prob`, the same iterations and loss), and the composites' moved by the stronger pull on π (the double ring's π 0.872 → 0.875, the cap with a ring's 0.859 → 0.861; every position within 0.02 nm); every tutorial fit still equals LocMoFit.jl's. `tests/common.py` and `julia_fits.jl` read the tutorial's `logit_prob` as the earlier parametrization. Speed (`benchmarks/speed/speed.py`, twice each against the previous commit, alternating): the compiled fits and the 20-site runs took the same time within run-to-run noise, with the same compilation classes; some float32 fits took another path (the ring 29 → 43 iterations, the cap with a ring 94 → 85).

**Callers and tutorial parameters (2026-11-17, as LocMoFit.jl's branch `priors`, its NOTES.md).** `run_locmofit.m` passes `params` (and finds the env in this folder). The tutorial's parameter files in `notebooks/data/` (copies of LocMoFit.jl's) give the truth as `alpha` and `omega0` instead of `eta` and `logit_prob`, so `tests/common.py` and `tests/data/julia_fits.jl` read only the current names; `julia_fits.json` was regenerated and moved by rounding only (at most 7e-12).

**The SMAPpy plugin (2026-11-17).** Ported by removing its *start values*, *freeze* and *tie* fields (the owner's choice for now): each model starts from `run_locmofit`'s own starting values, with every quantity free under its default prior; the model points still come from the default sizes (`point_params`). SMAPpy drops saved settings a plugin no longer has, so saved pipelines still load. Its `version` is now "2", since the fits' numbers changed (SMAPpy's batch reruns files whose steps' versions changed). Checked in SMAPpy's environment (its editable install of this working tree) by running the plugin on each tutorial site: all five models converged.

## Compilations kept on disk across SMAPpy sessions (2026-11-09)

**The choice.** The SMAPpy plugin switches on JAX's persistent compilation cache before its first fit (`use_compilation_cache`): the folder `locmofit_jax_cache` in SMAPpy's configuration folder, and a minimum compile time of 0, since JAX's default (1 s) would skip most of our compilations. A folder already set in `JAX_COMPILATION_CACHE_DIR` is kept. `run_locmofit` does not do it, because both settings hold for the whole process and a library should not change them for its caller; MATLAB or a script gets the same by setting `JAX_COMPILATION_CACHE_DIR` and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` in the environment.

**What it saves.** Measured through SMAPpy (`ROIProject.evaluate`, as *Evaluate* runs it, the plugin unchanged and the cache switched on by those two environment variables), JAX 0.12.0.1, Apple M1 Pro, CPU, 3D, float64. Each file holds 200 ROIs: LocMoFit.jl's 20 accuracy-benchmark sites (15% background), 10 copies each, every copy a random 35–100% of its localizations so that the sites fall into many size classes. Every column is a new process:

| 200 ROIs | No cache | Cache empty | Cache filled |
| --- | --- | --- | --- |
| Ring, whole run (3 repeats) | 17.7–19.4 s | 18.8–21.1 s | 8.8–9.2 s |
| Ring, XLA compiling | 11.1–12.3 s | 12.2–13.6 s | 2.3–2.4 s |
| SpCapRing, whole run (once) | 92.1 s | 97.6 s | 85.4 s |
| SpCapRing, XLA compiling | 16.1 s | 18.1 s | 3.7 s |

A second SpCapRing file (other copies of the sites, another grid), fitted in a new session with the cache the first file filled, compiled in 4.1 s instead of 16.7 s: 124 of its 176 compilations came from the cache, and the rest (size classes the first file had not reached) were added to it. The fitted values of every ROI were identical with and without the cache. Filling an empty cache costs about 1–2.5 s for the 200 rings (six interleaved pairs of sessions), once per JAX version (an update of JAX invalidates every entry). The two models' entries took 10 MB; JAX sets no size limit by default. So the saving is a fixed amount per session, about 0.6–0.9 s per setup and size class, and matters most for short sessions, quick fits and sites spread over many size classes; within a session the compiled fits are in memory anyway.

**What it cannot save.** The cache's key is a hash of the lowered program (with the compile options, the JAX version and the device), so JAX traces and lowers the fit in every session to find the entry; only the XLA compile becomes a load from disk. Per size class with the cache filled (`run_locmofit`, tutorial sites):

| | Trace | Lower | XLA compile → load |
| --- | --- | --- | --- |
| Ring | 0.21 s | 0.08 s | 0.78 → 0.16 s |
| SpCapRing | 0.38 s | 0.12 s | 1.17 → 0.24 s |

**Saving compiled programs ourselves, not done.** Serializing the executable (`jit(f).lower(...).compile()` and `jax.experimental.serialize_executable`) and loading it in a later session would skip tracing and lowering too, a further 0.3–0.5 s per size class (not measured). But the key would then be ours, built before tracing from what we know determines the program: the `FitStatic`, the argument shapes and dtypes, `max_iter`, `tol`, `history`, the JAX version, the device, and the version of LocMoFitPy's own code. An editable install changes the code without changing any version, and a key that misses a change runs the old program silently, an accuracy risk; it also needs code to write, find and prune the files, on an experimental API. JAX's own key is the program itself, so a stale entry can never be loaded. `jax.export` (the lowered program, which skips only tracing) has the same key problem.

**The folder.** SMAPpy's configuration folder, which `SMAPPY_CONFIG_DIR` redirects (so tests get their own), not LocMoFitPy's package folder: that belongs to the installer (every update from GitHub reinstalls it with `pip install --upgrade`), with an editable install it is the git checkout, and it may not be writable. The platform's cache folder (`~/Library/Caches`, `~/.cache`, `%LOCALAPPDATA%`) would be the proper place for files that can be rebuilt, but choosing it needs per-platform code or `platformdirs`, a new dependency; the configuration folder's drawback is that backups of the settings include the cache.

## Float32 fits can stop early (found 2026-10-29, accepted the same day)

As in LocMoFit.jl (its `NOTES.md`, known issue), BFGS can stop in float32 when the loss is unchanged in two successive iterations although the parameters still move (0.01–0.04 per step) and the largest gradient component is above `tol`: a mean loss of about 15 is resolved to about 1e-6 in float32. On 10 starts nudged by 1e-3 around the tutorial sites' moved start, fits stopped above `tol` on the ring 3, the cap 4 and the cap with ring 8 times out of 10 before task 2, and 8, 7 and 10 times after it; never on the double ring, and never in float64, the default. The loss then stays within about 1e-4 per localization of the optimum, and the parameters move by up to 0.5–0.9 (nm or the cap's units) against statistical errors of 1–12 nm.

**Not caused by task 2.** On one set of 10 nudged starts per site (ring, cap, cap with ring), the current code stopped early 0, 6 and 9 times; with the gradient from autodiff instead of `_ll_bwd` 1, 6 and 8; with the value evaluated apart from the gradient (the old line search) 3, 5 and 9; without padding 2, 5 and 9. Each change only moves the rounding along the path, so the counts vary with the start, not with the optimization.

**Far from the origin it is worse (measured 2026-11-13).** As in LocMoFit.jl (its `NOTES.md`, where the cause is described): float32 resolves a coordinate at 100 µm only to 0.0078 nm, and the model points are placed at the absolute coordinates. The accuracy benchmark's sites (LocMoFit.jl's, 20 per model, background, moved start) moved by 25 and 100 µm in x and y, against the float64 fit at the original position; each cell: the largest change of the position and of the axis, in units of the statistical error, and the median loss above the float64 optimum, per localization:

| | float32 at the origin | at 25 µm | at 100 µm |
| --- | --- | --- | --- |
| Ring | 0.02, 0.01; 4·10⁻¹⁰ | 0.04, 0.01; 6·10⁻⁷ | 0.23, 0.04; 2·10⁻⁶ |
| SphericalCap | 0.32, 0.31; 4·10⁻⁵ | 0.53, 0.50; 4·10⁻⁵ | 2.8, 2.7; 1.2·10⁻² |
| DoubleRing | 0.13, 0.13; 7·10⁻⁵ | 0.14, 0.12; 6·10⁻⁵ | 0.66, 0.47; 1·10⁻⁴ |
| SpCapRing | 0.34, 0.58; 1.4·10⁻⁴ | 0.38, 0.61; 6·10⁻⁵ | 0.43, 1.2; 8·10⁻⁴ |

float64 fits moved by 100 µm changed by at most 6·10⁻⁷ nm. The cap fits at 100 µm stop up to 13.6 nm and 6.9° from the optimum and report `Success`, as in LocMoFit.jl; the SMAPpy plugin fits in float64 unless float32 is chosen.

**Accepted (the owner's decision).** float64 is the default; float32 is faster (a cap about 1.7×) and may stop up to about 1e-4 per localization above the optimum. Possible remedies, not done and to be made in both packages: finish a float32 fit with a few float64 iterations from where it stopped (exact; here it needs `jax_enable_x64`, which float64 fits switch on anyway), or subtract each localization's log density at the start from the loss so that its changes keep more digits (on the same starts: 0, 1 and 3 early stops; it helps but does not end them), or subtract the site's position before the fit and add it back after, which would remove the loss of accuracy far from the origin but not the early stops at the origin.
