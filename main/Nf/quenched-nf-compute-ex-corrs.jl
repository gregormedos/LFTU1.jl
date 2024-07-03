using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ProgressBars
using Dates
import LinearAlgebra.dot
using ArgParse

parse_commandline() = parse_commandline(ARGS)
function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "-L"
        help = "lattice size"
        required = true
        arg_type = Int

        "--mass"
        help = "mass"
        required = true
        arg_type = Float64

        "--start"
        help = "start from configuration"
        required = false
        arg_type = Int
        default = 1

        "--nconf"
        help = "number of configurations to analyze; 0 means until the end"
        required = false
        arg_type = Int
        default = 0

        "--ens"
        help = "path to ensemble with configurations"
        required = true
        arg_type = String
        # default = "configs/"
    end
    return parse_args(args, s)
end

args = [
"-L", "20",
"--mass", "0.05",
"--ens", "01_beta5_lsize20_PBC_tau1_nsteps10/main.bdio",
"--start", "500",
"--nconf", "1000",
]
parsed_args = parse_commandline(args)

# parsed_args = parse_commandline(ARGS)


const NFL = 2  # number of flavors, hardcoded to be 2 by now
const N0 = parsed_args["L"]
const MASS = parsed_args["mass"]

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Quenched)

data = (
    # 2-point correlators
    P = zeros(Float64, NFL, NFL, N0),
    disc = zeros(Float64, NFL, NFL, N0),
    # for bulding 2-point correlators
    Delta = zeros(Float64, NFL, N0),
    # 4-point correlators
    P4_tt0t0t = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    P4_tt0tt0 = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    P4_1DeltaInitial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    P4_1DeltaFinal = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    P4_2P = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    P4_2Delta = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2PSpatial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2DeltaSpatial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_4Delta = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    # for bulding 4-point correlators
    PSpatial = zeros(Float64, NFL, NFL, N0),
    P_2DeltaSpatial = zeros(Float64, NFL, NFL, N0)
)
Data = typeof(data)

function reset_data(data::Data)
    data.P .= 0.0
    data.disc .= 0.0
    data.Delta .= 0.0
    data.P4_tt0t0t .= 0.0
    data.P4_tt0t0t .= 0.0
    data.P4_1DeltaInitial .= 0.0
    data.P4_1DeltaFinal .= 0.0
    data.P4_2P .= 0.0
    data.P4_2Delta .= 0.0
    data.disc4_2PSpatial .= 0.0
    data.disc4_2DeltaSpatial .= 0.0
    data.disc4_4Delta .= 0.0
    data.PSpatial .= 0.0
    data.P_2DeltaSpatial .= 0.0
    return nothing
end

"""
- Compute connected traces and save them into data.P[ifl, jfl, t]
- Compute disconnected traces separately and save them to data.Delta[ifl, t]
- Compute connected 4-point traces and save them into data.P4_tt0t0t[ifl1, ifl2, ifl3, ifl4, t] or data.P4_tt0tt0[ifl1, ifl2, ifl3, ifl4, t]
- Compute initially connected 4-point traces and save them into data.P4_1DeltaFinal[ifl1, ifl2, ifl3, ifl4, t]
- Compute finally connected 4-point traces and save them into data.P4_1DeltaInitial[ifl1, ifl2, ifl3, ifl4, t]
- Compute partially connected 4-point traces with 2 P traces and save them into data.P4_2P[ifl1, ifl2, ifl3, ifl4, t]
- Compute partially connected 4-point traces with 2 Delta traces and save them into data.P4_2Delta[ifl1, ifl2, ifl3, ifl4, t]
- Compute disconnected P traces and save them into data.PSpatial[ifl, jfl, t]
- Compute disconnected 2-point traces with 2 connected Delta traces and save them into data.P_2DeltaSpatial[ifl, jfl, t]
"""
function correlators(data::Data, corrws::U1exCorrelator, u1ws)
    reset_data(data)
    for ifl in 1:NFL
        ex_disconnected_correlator(corrws, u1ws, ifl)
        data.Delta[ifl,:] .+= corrws.result
        for jfl in 1:NFL
            ex_spatially_connected_correlator(corrws, u1ws, ifl, jfl)
            data.PSpatial[ifl,jfl,:] .+= corrws.result
            for it in 1:N0
                ex_connected_correlator(corrws, u1ws, it, ifl, jfl)
                data.P[ifl,jfl,it] += corrws.result[it]
            end
            for ifl3 in 1:NFL
                for it in 1:N0
                    ex_3point_initially_connected_correlator(corrws, u1ws, it, ifl, jfl, ifl3)
                    data.P3Initial[ifl,jfl,ifl3,it] += corrws.result[it]
                    ex_3point_finally_connected_correlator(corrws, u1ws, it, ifl, jfl, ifl3)
                    data.P3Final[ifl,jfl,ifl3,it] += corrws.result[it]
                end
                for ifl4 in 1:NFL
                    for it in 1:N0
                        ex_4point_connected_correlator_tt0t0t(corrws, u1ws, it, ifl, jfl, ifl3, ifl4)
                        data.P4_tt0t0t[ifl,jfl,ifl3,ifl4,it] += corrws.result[it]
                        ex_4point_connected_correlator_tt0tt0(corrws, u1ws, it, ifl, jfl, ifl3, ifl4)
                        data.P4_tt0tt0[ifl,jfl,ifl3,ifl4,it] += corrws.result[it]
                    end
                end
            end
        end
    end
    compute_disconnected!(data)
    compute_disconnected4!(data)
    average_over_initial_times(data)
    return nothing
end

function average_over_initial_times(data)
    result = copy(data.P)
    data.P .= 0.0
    for t in 1:N0
        tt=((t-it+N0)%N0+1);
        data.P[:,:,tt] += result[:,:,t] / N0
    end

    result = copy(data.P4_tt0t0t)
    data.P4_tt0t0t .= 0.0
    for t in 1:N0
        tt=((t-it+N0)%N0+1);
        data.P4_tt0t0t[:,:,:,:,tt] += result[:,:,:,:,t] / N0
    end

    result = copy(data.P4_tt0tt0)
    data.P4_tt0tt0 .= 0.0
    for t in 1:N0
        tt=((t-it+N0)%N0+1);
        data.P4_tt0tt0[:,:,:,:,tt] += result[:,:,:,:,t] / N0
    end

    result = copy(data.P4_1DeltaInitial)
    data.P4_1DeltaInitial .= 0.0
    for t in 1:N0
        tt=((t-it+N0)%N0+1);
        data.P4_1DeltaInitial[:,:,:,:,tt] += result[:,:,:,:,t] / N0
    end

    result = copy(data.P4_1DeltaFinal)
    data.P4_1DeltaFinal .= 0.0
    for t in 1:N0
        tt=((t-it+N0)%N0+1);
        data.P4_1DeltaFinal[:,:,:,:,tt] += result[:,:,:,:,t] / N0
    end

    return nothing
end

"""
Compute all combinations of disconnected traces and save them to data.disc[ifl, jfl, t]
"""
function compute_disconnected!(data::Data)
    data.disc .= 0.0
    for ifl in 1:NFL, jfl in ifl:NFL
        for t in 1:N0, tt in 1:N0
            data.disc[ifl, jfl, t] += data.Delta[ifl, tt] * data.Delta[jfl, (tt+t-1-1)%N0+1] / N0
        end
    end
    return nothing
end

"""
Compute all combinations of disconnected traces and save them to:
- data.disc4_1DeltaInitial[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_1DeltaFinal[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2P[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2PSpatial[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2Delta[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2DeltaSpatial[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_4Delta[ifl, ifl2, ifl3, ifl4, t]
"""
function compute_disconnected4!(data::Data)
    data.disc4_1DeltaInitial .= 0.0
    data.disc4_1DeltaFinal .= 0.0
    data.disc4_2P .= 0.0
    data.disc4_2PSpatial .= 0.0
    data.disc4_2Delta .= 0.0
    data.disc4_2DeltaSpatial .= 0.0
    data.disc4_4Delta .= 0.0
    for ifl1 in 1:NFL
        for ifl2 in 1:NFL, ifl3 in ifl2:NFL, ifl4 in ifl3:NFL
            for t in 1:N0, tt in 1:N0
                data.disc4_1DeltaInitial[ifl1, ifl2, ifl3, ifl4, t] += data.Delta[ifl1, tt] * data.P3Final[ifl2, ifl3, ifl4, (tt+t-1-1)%N0+1] / N0
                data.disc4_1DeltaFinal[ifl1, ifl2, ifl3, ifl4, t] += data.Delta[ifl1, tt] * data.P3Initial[ifl2, ifl3, ifl4, (tt+t-1-1)%N0+1] / N0
            end
        end
        for ifl2 in ifl1:NFL
            for ifl3 in 1:NFL, ifl4 in ifl3:NFL
                for t in 1:N0, tt in 1:N0
                    data.disc4_2P[ifl1, ifl2, ifl3, ifl4, t] += data.P[ifl1, ifl2, tt] * data.P[ifl2, ifl3, ifl4, (tt+t-1-1)%N0+1] / N0
                    data.disc4_2PSpatial[ifl1, ifl2, ifl3, ifl4, t] += data.Delta[ifl1, tt] * data.P3Initial[ifl2, ifl3, ifl4, (tt+t-1-1)%N0+1] / N0
                end
            end
        end
    end
    return nothing
end

function save_data(data::Data, dirpath)
    for ifl in 1:NFL
        deltafile = joinpath(dirpath, "measurements/exdelta-$(ifl)_confs$start-$finish.txt")
        write_vector(data.Delta[ifl, :],deltafile)
        for jfl in ifl:NFL
            connfile = joinpath(dirpath,"measurements/exconn-$ifl$(jfl)_confs$start-$finish.txt")
            discfile = joinpath(dirpath, "measurements/exdisc-$ifl$(jfl)_confs$start-$finish.txt")
            write_vector(data.P[ifl, jfl, :],connfile)
            write_vector(data.disc[ifl, jfl, :],discfile)
        end
    end
end

function save_topcharge(model, dirpath)
    qfile = joinpath(dirpath,"measurements/topcharge_confs$start-$finish.txt")
    global io_stat = open(qfile, "a")
    write(io_stat, "$(top_charge(model))\n")
    close(io_stat)
end

function write_vector(vec, filepath)
    global io_stat = open(filepath, "a")
    write(io_stat, "$(vec[1])")
    for i in 2:length(vec)
        write(io_stat, ",$(vec[i])")
    end
    write(io_stat, "\n")
    close(io_stat)
    return nothing
end


pws = U1exCorrelator(model, wdir=dirname(cfile))
for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    construct_invgD!(pws, model, MASS)
    correlators(data, pws, model)
    save_data(data, dirname(cfile))
    save_topcharge(model, dirname(cfile))
end
close(fb)

