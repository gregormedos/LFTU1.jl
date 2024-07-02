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
    # for building 2-point correlators
    P = zeros(Float64, NFL, NFL, N0),
    disc = zeros(Float64, NFL, NFL, N0),
    Delta = zeros(Float64, NFL, N0),
    # for bulding 4-point correlators
    P4 = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_1DeltaInitial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_1DeltaFinal = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2PDeltas = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2PDeltasSpatial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2Deltas = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_2DeltasSpatial = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    disc4_4Deltas = zeros(Float64, NFL, NFL, NFL, NFL, N0),
    PDelta = zeros(Float64, NFL, NFL, N0),
    P3Initial = zeros(Float64, NFL, NFL, NFL, N0),
    P3Final = zeros(Float64, NFL, NFL, NFL, N0)
)
Data = typeof(data)

function reset_data(data::Data)
    data.P .= 0.0
    data.disc .= 0.0
    data.Delta .= 0.0
    data.P4 .= 0.0
    data.disc4_1DeltaInitial .= 0.0
    data.disc4_1DeltaFinal .= 0.0
    data.disc4_2PDeltas .= 0.0
    data.disc4_2PDeltasSpatial .= 0.0
    data.disc4_2Deltas .= 0.0
    data.disc4_2DeltasSpatial .= 0.0
    data.disc4_4Deltas .= 0.0
    data.PDelta .= 0.0
    data.P3Initial .= 0.0
    data.P3Final .= 0.0
    return nothing
end

"""
- Compute connected traces and save them into data.P[ifl, jfl, t]
- Compute disconnected traces separately and save them to data.Delta[ifl, t]
- Compute spatially connected but temporally disconnected traces separately and save them into data.PDelta[ifl, jfl, t]
- Compute initially connected 3-point traces separately and save them into data.P3Initial[ifl1, ifl2, ifl3, t]
- Compute finally connected 3-point traces separately and save them into data.P3Final[ifl1, ifl2, ifl3, t]
- Compute connected 4-point traces and save them into data.P4[ifl1, ifl2, ifl3, ifl4, t]
""" 
function correlators(data::Data, corrws::U1exCorrelator, u1ws)
    reset_data(data)
    for ifl in 1:NFL
        ex_disconnected_correlator(corrws, u1ws, ifl)
        data.Delta[ifl,:] .+= corrws.result
        for jfl in 1:NFL
            for it in 1:N0
                ex_connected_correlator(corrws, u1ws, it, ifl, jfl)
                for t in 1:N0
                    tt=((t-it+N0)%N0+1);
                    data.P[ifl,jfl,tt] += corrws.result[t] / N0
                end
            ex_spatially_connected_correlator(corrws, u1ws, ifl, jfl)
            data.PDelta[ifl,jfl,:] .+= corrws.result
            end
            for ifl3 in 1:NFL
                for it in 1:N0
                    ex_3point_initially_connected_correlator(corrws, u1ws, it, ifl, jfl, ifl3)
                    for t in 1:N0
                        tt=((t-it+N0)%N0+1);
                        data.P3Initial[ifl,jfl,ifl3,tt] += corrws.result[t] / N0
                    end
                    ex_3point_finally_connected_correlator(corrws, u1ws, it, ifl, jfl, ifl3)
                    for t in 1:N0
                        tt=((t-it+N0)%N0+1);
                        data.P3Final[ifl,jfl,ifl3,tt] += corrws.result[t] / N0
                    end
                end
                for ifl4 in 1:NFL
                    for it in 1:N0
                        ex_4point_connected_correlator(corrws, u1ws, it, ifl, jfl, ifl3, ifl4)
                        for t in 1:N0
                            tt=((t-it+N0)%N0+1);
                            data.P4[ifl,jfl,ifl3,ifl4,tt] += corrws.result[t] / N0
                        end
                    end
                end
            end
        end
    end
    compute_disconnected!(data)
    compute_disconnected4!(data)
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
- data.disc4_2PDeltas[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2PDeltasSpatial[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2Deltas[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_2DeltasSpatial[ifl, ifl2, ifl3, ifl4, t]
- data.disc4_4Deltas[ifl, ifl2, ifl3, ifl4, t]
"""
function compute_disconnected4!(data::Data)
    # TODO: 7 types of combinations
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

