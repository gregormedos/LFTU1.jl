using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using Printf
using Statistics
using Plots
default(legend = false)
using LaTeXStrings
import DelimitedFiles
import ADerrors
import LsqFit


function read_trace_timeslices(filename)
    trace_timeslices = Vector{ADerrors.uwreal}(undef, lsize)
    filepath = joinpath(@__DIR__, "measurements", filename*".txt")
    data = DelimitedFiles.readdlm(filepath, ',')
    for t in 1:lsize
        trace_timeslices[t] = ADerrors.uwreal(data[:,t], "$filename at time $t")
    end
    return trace_timeslices
end


function get_pion_correlation_function(traces_timeslices)
    exconn11_timeslices = traces_timeslices[1]
    for t in 1:lsize
        ADerrors.uwerr(exconn11_timeslices[t])
    end
    return exconn11_timeslices
end


function get_etaprime_correlation_function(traces_timeslices)
    exconn11_timeslices = traces_timeslices[1]
    exdisc11_timeslices = traces_timeslices[2]
    result = exconn11_timeslices .- 2 .* exdisc11_timeslices
    for t in 1:lsize
        ADerrors.uwerr(result[t])
    end
    return result
end


function get_2pion_isospin2_channel_correlation_function(traces_timeslices)
    exD1111_timeslices = traces_timeslices[1]
    exC1111_timeslices = traces_timeslices[2]
    result = 2 * (exD1111_timeslices - exC1111_timeslices)
    for t in 1:lsize
        ADerrors.uwerr(result[t])
    end
    return result
end


function effective_mass_function(corr)
    result = Vector{ADerrors.uwreal}(undef, lsize-2)
    for tt in 1:lsize-2
        result[tt] = (corr[tt] + corr[tt+2]) / (2 * corr[tt+1])
        result[tt] = acosh(result[tt])
        ADerrors.uwerr(result[tt])
    end
    return result
end


function plot_correlation_function(name, get_correlation_function, traces_timeslices)
    corr = get_correlation_function(traces_timeslices)
    expval_corr = Vector{Float64}(undef, lsize)
    error_corr = Vector{Float64}(undef, lsize)
    for t in 1:lsize
        expval_corr[t] = ADerrors.value(corr[t])
        error_corr[t] = ADerrors.err(corr[t])
    end
    plot(temporal_sites, expval_corr, yerror=error_corr, xlabel=L"t", ylabel=L"C(t)")
    savefig(joinpath(@__DIR__, "$(name)_corr.svg"))

    correlator_fit_function(t, p) = p[1] .* cosh.(p[2].*(t.-lsize./2))
    p0 = [1.0, 1.0]
    fit_to_data = LsqFit.curve_fit(correlator_fit_function, temporal_sites, expval_corr, p0)
    p = fit_to_data.param
    plot(temporal_sites, expval_corr, xlabel=L"t", ylabel=L"C(t)", label="HMC", title="The fitted mass: $(@sprintf("%.3f", p[2]))")
    plot!(temporal_sites, correlator_fit_function(temporal_sites, p), linestyle=:dash, label="fit", legend=:top)
    savefig(joinpath(@__DIR__, "$(name)_corr_fit.svg"))

    effective_mass = effective_mass_function(corr)
    expval_effective_mass = Vector{Float64}(undef, lsize-2)
    error_effective_mass = Vector{Float64}(undef, lsize-2)
    for t in 1:lsize-2
        expval_effective_mass[t] = ADerrors.value(effective_mass[t])
        error_effective_mass[t] = ADerrors.err(effective_mass[t])
    end
    plot(temporal_sites[2:lsize-1], expval_effective_mass, yerror=error_effective_mass, xlabel=L"t", ylabel=L"m_\mathrm{eff}(t)", title="The effective mass: "*@sprintf("%.3f", expval_effective_mass[lsize÷2])*L"\pm"*@sprintf("%.3f", error_effective_mass[lsize÷2]))
    savefig(joinpath(@__DIR__, "$(name)_eff-mass.svg"))
end


# Configurations
filename = "main"
fb, model = read_cnfg_info(joinpath(@__DIR__, filename*".bdio"), U1Quenched)
lsize = model.params.iL[1]
temporal_sites = [1:lsize;]


exconn11_timeslices = read_trace_timeslices("exconn-11_confs500-1499")
exdisc11_timeslices = read_trace_timeslices("exdisc-11_confs500-1499")

plot_correlation_function("pion", get_pion_correlation_function, [exconn11_timeslices])
plot_correlation_function("etaprime", get_etaprime_correlation_function, [exconn11_timeslices, exdisc11_timeslices])


exD1111_timeslices = read_trace_timeslices("exD-1111_confs500-1499")
exC1111_timeslices = read_trace_timeslices("exC-1111_confs500-1499")

plot_correlation_function("pion_isospin2_channel", get_2pion_isospin2_channel_correlation_function, [exD1111_timeslices, exC1111_timeslices])
