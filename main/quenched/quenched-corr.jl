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
import DelimitedFiles
import LsqFit


function read_trace_timeslices(filename)
    expval_trace_timeslices = zeros(Float64, lsize)
    error_trace_timeslices = zeros(Float64, lsize)
    uncertain_trace_timeslices = zeros(Measurements.Measurement, lsize)
    filepath = joinpath(@__DIR__, "measurements", filename*".txt")
    data = DelimitedFiles.readdlm(filepath, ',')
    for t in 1:lsize
        # Block average
        expval, error = block_average(data[:, t], 20)
        expval_trace_timeslices[t] = expval
        error_trace_timeslices[t] = error
        uncertain_trace_timeslices[t] = expval ± error
    end
    return expval_trace_timeslices, error_trace_timeslices, uncertain_trace_timeslices
end


function get_pion_correlation_function(exconn11_timeslice)
    return exconn11_timeslice
end


function get_etaprime_correlation_function(exconn11_timeslice, exdisc11_timeslice)
    return 2 * exdisc11_timeslice - exconn11_timeslice
end


function effective_mass_function(corr_behind, corr, corr_forward)
    return acosh((corr_forward + corr_behind) / (2 * corr))
end


expval_exconn11_timeslices, error_exconn11_timeslices, uncertain_exconn11_timeslices = read_trace_timeslices("exconn-11_confs500-1499")
expval_exdisc11_timeslices, error_exdisc11_timeslices, uncertain_exdisc11_timeslices = read_trace_timeslices("exdisc-11_confs500-1499")

expval_pion_corr = zeros(Float64, lsize)
error_pion_corr = zeros(Float64, lsize)
uncertain_pion_corr = zeros(Measurements.Measurement, lsize)
for t in 1:lsize
    uncertain_pion_corr[t] = Measurements.@uncertain get_pion_correlation_function(uncertain_exconn11_timeslices[t])
    expval_pion_corr[t] = Measurements.value(uncertain_pion_corr[t])
    error_pion_corr[t] = Measurements.uncertainty(uncertain_pion_corr[t])
end
plot(temporal_sites, expval_pion_corr, yerror=error_pion_corr, xlabel=L"t", ylabel=L"C(t)")
savefig(joinpath(@__DIR__, "pion_corr.svg"))

correlator_fit_function(t, p) = p[1] .* cosh.(p[2].*(t.-lsize./2))
p0 = [1.0, 1.0]
fit_to_data = LsqFit.curve_fit(correlator_fit_function, temporal_sites, expval_pion_corr, p0)
p = fit_to_data.param
plot(temporal_sites, expval_pion_corr, xlabel=L"t", ylabel=L"C(t)", label="HMC", title="The fitted mass of the pion: $(@sprintf("%.3f", p[2]))")
plot!(temporal_sites, correlator_fit_function(temporal_sites, p), linestyle=:dash, label="fit", legend=:top)
savefig(joinpath(@__DIR__, "pion_corr_fit.svg"))

expval_effective_mass = zeros(Float64, lsize-2)
error_effective_mass = zeros(Float64, lsize-2)
uncertain_effective_mass = zeros(Measurements.Measurement, lsize-2)
for i in 1:lsize-2
    uncertain_effective_mass[i] = Measurements.@uncertain effective_mass_function(uncertain_pion_corr[i], uncertain_pion_corr[i+1], uncertain_pion_corr[i+2])
    expval_effective_mass[i] = Measurements.value(uncertain_effective_mass[i])
    error_effective_mass[i] = Measurements.uncertainty(uncertain_effective_mass[i])
end
plot(temporal_sites[2:end-1], expval_effective_mass, yerror=error_effective_mass, xlabel=L"t", ylabel=L"m_\mathrm{eff}(t)", title="The effective mass of the pion: "*@sprintf("%.3f", expval_effective_mass[lsize÷2])*L"\pm"*@sprintf("%.3f", error_effective_mass[lsize÷2]))
savefig(joinpath(@__DIR__, "pion_eff-mass.svg"))

expval_etaprime_corr = zeros(Float64, lsize)
error_etaprime_corr = zeros(Float64, lsize)
uncertain_etaprime_corr = zeros(Measurements.Measurement, lsize)
for t in 1:lsize
    uncertain_etaprime_corr[t] = Measurements.@uncertain get_etaprime_correlation_function(uncertain_exconn11_timeslices[t], uncertain_exdisc11_timeslices[t])
    expval_etaprime_corr[t] = Measurements.value(uncertain_etaprime_corr[t])
    error_etaprime_corr[t] = Measurements.uncertainty(uncertain_etaprime_corr[t])
end
plot(temporal_sites, expval_etaprime_corr, yerror=error_etaprime_corr, xlabel=L"t", ylabel=L"C(t)")
savefig(joinpath(@__DIR__, "etaprime_corr.svg"))

correlator_fit_function(t, p) = p[1] .* cosh.(p[2].*(t.-lsize./2))
p0 = [1.0, 1.0]
fit_to_data = LsqFit.curve_fit(correlator_fit_function, temporal_sites, expval_etaprime_corr, p0)
p = fit_to_data.param
plot(temporal_sites, expval_etaprime_corr, xlabel=L"t", ylabel=L"C(t)", label="HMC", title="The fitted mass of the eta prime: $(@sprintf("%.3f", p[2]))")
plot!(temporal_sites, correlator_fit_function(temporal_sites, p), linestyle=:dash, label="fit", legend=:top)
savefig(joinpath(@__DIR__, "etaprime_corr_fit.svg"))

expval_effective_mass = zeros(Float64, lsize-2)
error_effective_mass = zeros(Float64, lsize-2)
uncertain_effective_mass = zeros(Measurements.Measurement, lsize-2)
for i in 1:lsize-2
    uncertain_effective_mass[i] = Measurements.@uncertain effective_mass_function(uncertain_etaprime_corr[i], uncertain_etaprime_corr[i+1], uncertain_etaprime_corr[i+2])
    expval_effective_mass[i] = Measurements.value(uncertain_effective_mass[i])
    error_effective_mass[i] = Measurements.uncertainty(uncertain_effective_mass[i])
end
plot(temporal_sites[2:end-1], expval_effective_mass, yerror=error_effective_mass, xlabel=L"t", ylabel=L"m_\mathrm{eff}(t)", title="The effective mass of the eta prime: "*@sprintf("%.3f", expval_effective_mass[lsize÷2])*L"\pm"*@sprintf("%.3f", error_effective_mass[lsize÷2]))
savefig(joinpath(@__DIR__, "etaprime_eff-mass.svg"))
