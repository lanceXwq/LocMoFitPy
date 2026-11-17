# LocMoFit.jl's fits of the tutorial data (notebooks/data, copies of LocMoFit.jl's examples/data) from the moved start of LocMoFit.jl's test/fit.jl, in Float64, and its starting values estimated from the same data (estimate_start): the references for the parity tests of tests/test_fit.py and tests/test_start.py. Run from the LocMoFitPy folder:
#     julia --project=../LocMoFit.jl tests/data/julia_fits.jl
# Writes tests/data/julia_fits.json and tests/data/julia_starts.json.
using Printf
# readcsv, columns, matrix, moved and start_params, as LocMoFit.jl's tests and benchmarks use them
include(joinpath(@__DIR__, "..", "..", "..", "LocMoFit.jl", "test", "tutorial.jl"))

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")
entries, starts = String[], String[]
for (case, model, D) in (("ring", "Ring", 3), ("ring2D", "Ring", 2), ("double_ring", "DoubleRing", 3),
                         ("spcap", "SphericalCap", 3), ("spcap_ring", "SpCapRing", 3))
    d = readcsv(joinpath(DATA, "$case.csv"))
    t = map(first, readcsv(joinpath(DATA, "$(case)_parameters.csv")))
    cols = columns(D)
    locs, precs = matrix(d, cols[1]), matrix(d, cols[2])
    init, fixed = moved(t), D == 2 ? (:z,) : ()
    D == 2 && (init = merge(init, (z=t.z,)))
    e = LocMoFit.estimate_start(LocMoFit.build_model(model), locs, precs, nothing)
    push!(starts, @sprintf("  \"%s\": {%s}", case, join([@sprintf("\"%s\": %.17g", k, v) for (k, v) in pairs(e)], ", ")))
    r = run_locmofit(model, locs, precs; params=start_params(init; fixed), dtype=Float64, fov_volume=300.0^D)
    params = join([@sprintf("\"%s\": %.17g", k, v) for (k, v) in pairs(r.parameters)], ", ")
    push!(entries, @sprintf("  \"%s\": {\"iterations\": %d, \"loss\": %.17g, \"parameters\": {%s}}", case,
                            r.result.stats.iterations, r.loss, params))
end
open(joinpath(@__DIR__, "julia_fits.json"), "w") do io
    println(io, "{\n", join(entries, ",\n"), "\n}")
end
open(joinpath(@__DIR__, "julia_starts.json"), "w") do io
    println(io, "{\n", join(starts, ",\n"), "\n}")
end
