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

# Look at the action and other observables to decide
thermalization_steps = 500

# Configurations
filename = "main"


function calculate_observable_ensemble(observable, ens)
    observable_ensemble = zeros(Float64, length(ens))
    for i in eachindex(ens)
        observable_ensemble[i] = observable(ens[i])
    end
    return observable_ensemble
end


function block_average(observable_ensemble, nblocks)
    # Block average
    observable_nblocks = zeros(Float64, nblocks)
    chain_length = length(observable_ensemble)
    block_length = chain_length ÷ nblocks
    for i in 1:nblocks
        observable_nblocks[i] = mean(observable_ensemble[1+(i-1)*block_length:i*block_length])
    end
    observable_mean = mean(observable_nblocks)
    observable_std = std(observable_nblocks)
    observable_error = observable_std / sqrt(nblocks)
    return observable_mean, observable_error
end


ens = LFTSampling.read_ensemble(joinpath(@__DIR__, filename*".bdio"), U1Quenched)
println()

S = calculate_observable_ensemble(action, ens)
Q = calculate_observable_ensemble(top_charge, ens)
Q2 = Q.^2
fid = HDF5.h5open(joinpath(@__DIR__, "data.h5"), "w")
fid["action"] = S
fid["top_charge"] = Q
fid["top_charge_squared"] = Q2
close(fid)

fid = HDF5.h5open(joinpath(@__DIR__, "data.h5"), "r")
S = HDF5.read(fid["action"])
Q = HDF5.read(fid["top_charge"])
Q2 = HDF5.read(fid["top_charge_squared"])
close(fid)
plot(S, xlabel="Configuration sample", ylabel=L"S")
savefig(joinpath(@__DIR__, "action.svg"))
plot(Q, xlabel="Configuration sample", ylabel=L"Q")
savefig(joinpath(@__DIR__, "top_charge.svg"))
plot(Q2, xlabel="Configuration sample", ylabel=L"Q^2")
savefig(joinpath(@__DIR__, "top_charge_squared.svg"))
plot(S[1:500], xlabel="Configuration sample", ylabel=L"S")
savefig(joinpath(@__DIR__, "thermalization.svg"))

fid = HDF5.h5open(joinpath(@__DIR__, "data.h5"), "r")
S = HDF5.read(fid["action"])[thermalization_steps:end]
Q = HDF5.read(fid["top_charge"])[thermalization_steps:end]
Q2 = HDF5.read(fid["top_charge_squared"])[thermalization_steps:end]
close(fid)
B_max = 500
B = collect(1:1:B_max)
B_length = zeros(Int, B_max)
expval_S = zeros(Float64, B_max)
error_S = zeros(Float64, B_max)
expval_Q = zeros(Float64, B_max)
error_Q = zeros(Float64, B_max)
expval_Q2 = zeros(Float64, B_max)
error_Q2 = zeros(Float64, B_max)
for i in 1:B_max
    B_length[i] = length(S) ÷ B[i]
    expval_S[i], error_S[i] = block_average(S, B[i])
    expval_Q[i], error_Q[i] = block_average(Q, B[i])
    expval_Q2[i], error_Q2[i] = block_average(Q2, B[i])
end
plot(B_length[50:end], expval_S[50:end], xlabel=L"M_b", ylabel=L"<S>")
savefig(joinpath(@__DIR__, "S.svg"))
plot(B_length[50:end], error_S[50:end], xlabel=L"M_b", ylabel=L"\sigma_S/\sqrt{M_b}")
savefig(joinpath(@__DIR__, "S_error.svg"))
plot(B_length[50:end], expval_Q[50:end], xlabel=L"M_b", ylabel=L"<Q>")
savefig(joinpath(@__DIR__, "Q.svg"))
plot(B_length[50:end], error_Q[50:end], xlabel=L"M_b", ylabel=L"\sigma_Q/\sqrt{M_b}")
savefig(joinpath(@__DIR__, "Q_error.svg"))
plot(B_length[50:end], expval_Q2[50:end], xlabel=L"M_b", ylabel=L"<Q^2>")
savefig(joinpath(@__DIR__, "Q2.svg"))
plot(B_length[50:end], error_Q2[50:end], xlabel=L"M_b", ylabel=L"\sigma_{Q^2}/\sqrt{M_b}")
savefig(joinpath(@__DIR__, "Q2_error.svg"))
