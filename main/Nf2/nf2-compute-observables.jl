using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using Printf
using Statistics
using Measurements
using Plots
default(legend = false)
using LaTeXStrings
import HDF5


function calculate_observable_ensemble(observable, ens)
    observable_ensemble = zeros(Float64, length(ens))
    for i in eachindex(ens)
        observable_ensemble[i] = observable(ens[i])
    end
    return observable_ensemble
end


# Look at the action and other observables to decide
thermalization_steps = 100

# Configurations
filename = "main"
ens = LFTSampling.read_ensemble(joinpath(@__DIR__, filename*".bdio"), U1Quenched)
println()

S = calculate_observable_ensemble(action, ens)
Q = calculate_observable_ensemble(top_charge, ens)
Q2 = Q.^2

plot(S, xlabel="Configuration sample", ylabel=L"S")
savefig(joinpath(@__DIR__, "action.svg"))
plot(Q, xlabel="Configuration sample", ylabel=L"Q")
savefig(joinpath(@__DIR__, "top_charge.svg"))
plot(Q2, xlabel="Configuration sample", ylabel=L"Q^2")
savefig(joinpath(@__DIR__, "top_charge_squared.svg"))
plot(S[1:thermalization_steps], xlabel="Configuration sample", ylabel=L"S")
savefig(joinpath(@__DIR__, "thermalization.svg"))
