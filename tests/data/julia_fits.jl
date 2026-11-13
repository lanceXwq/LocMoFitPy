# LocMoFit.jl's fits of the tutorial data (notebooks/data, copies of LocMoFit.jl's examples/data) from the moved start of LocMoFit.jl's test/fit.jl, in Float64, and its starting values estimated from the same data (estimate_start): the references for the parity tests of tests/test_fit.py and tests/test_start.py. Run from the LocMoFitPy folder:
#     julia --project=../LocMoFit.jl tests/data/julia_fits.jl
# Writes tests/data/julia_fits.json and tests/data/julia_starts.json.
using LocMoFit, Printf

parsefield(s) = s == "true" ? 1.0 : s == "false" ? 0.0 : parse(Float64, s)
function readcsv(path)
    L = readlines(path)
    names = Tuple(Symbol.(split(L[1], ",")))
    rows = [parsefield.(split(l, ",")) for l in L[2:end]]
    return NamedTuple{names}(Tuple([r[k] for r in rows] for k in eachindex(names)))
end
function moved(p)
    shift(k, v) = k in (:x, :y, :z, :x1, :y1, :z1, :x2, :y2, :z2) ? v + 5 :
                  k in (:r, :r1, :r2, :s, :s2) ? 1.15v : k in (:eta, :eta2) ? atanh(0.85tanh(v)) :
                  k in (:theta, :phi, :theta1, :phi1, :theta2, :phi2) ? v + 0.15 : k == :sigma ? 2.0 :
                  k == :logit_prob ? log(9) : k == :omega2 ? 0.0 : v
    return NamedTuple{keys(p)}(Tuple(shift(k, v) for (k, v) in pairs(p)))
end

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")
entries, starts = String[], String[]
for (case, model, D) in (("ring", "Ring", 3), ("ring2D", "Ring", 2), ("double_ring", "DoubleRing", 3),
                         ("spcap", "SphericalCap", 3), ("spcap_ring", "SpCapRing", 3))
    d = readcsv(joinpath(DATA, "$case.csv"))
    t = map(first, readcsv(joinpath(DATA, "$(case)_parameters.csv")))
    cols = D == 3 ? ((:x_nm, :y_nm, :z_nm), (:x_err_nm, :y_err_nm, :z_err_nm)) : ((:x_nm, :y_nm), (:x_err_nm, :y_err_nm))
    locs = permutedims(reduce(hcat, [d[c] for c in cols[1]]))
    precs = permutedims(reduce(hcat, [d[c] for c in cols[2]]))
    init, freeze = moved(t), D == 2 ? (:z,) : ()
    D == 2 && (init = merge(init, (z=t.z,)))
    e = LocMoFit.estimate_start(LocMoFit.build_model(model), locs, precs, nothing)
    push!(starts, @sprintf("  \"%s\": {%s}", case, join([@sprintf("\"%s\": %.17g", k, v) for (k, v) in pairs(e)], ", ")))
    r = run_locmofit(model, locs, precs; init_params=init, freeze, dtype=Float64, fov_volume=300.0^D)
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
