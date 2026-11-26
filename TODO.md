# To-do

LocMoFitPy follows LocMoFit.jl: a change is made there first and ported here (`JULIA_DIFFERENCES.md` lists what is not ported yet), so LocMoFit.jl's `TODO.md` applies here too. This file lists what is specific to LocMoFitPy.

## Major

### Accuracy

### Performance

### Simplicity

### Extensibility

1. **SMAPpy plugin: a table of parameters.** Enter starting values as a table of physical quantities (labels and units, start, free, fixed or the same as another), as in MATLAB LocMoFit, instead of raw names. Open decisions: a generic table field in SMAPpy or its list field, and moving the update check to when the panel first opens, so that the form can read the model's parameters.
2. **SMAPpy plugin: build a model in the interface** by composing components, not only by choosing from `Ring`, `SphericalCap`, `DoubleRing` and `SpCapRing` (`Composite` already exists in code).
3. **SMAPpy plugin: start from an earlier step's fit** of the same ROI.
4. **SMAPpy plugin: the result summary.** *Show result* after a run over many ROIs says only how many ROIs and steps ran; decide what it should show (likely a change in SMAPpy).

## Minor

1. PEP 8 names for the math variables (`X`, `R`, `D`, `M`, …), deferred until parity is settled.
2. An Apple GPU (Metal) path, set aside on 2026-10-30: jax-metal does not work with current JAX, has no float64, and wins only on very large sites. Revisit if that changes.
