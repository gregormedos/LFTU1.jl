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
        help = "lattice spatial size"
        required = true
        arg_type = Int

        "-T"
        help = "lattice temporal size"
        required = true
        arg_type = Int

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
"-L", "40",
"-T", "48",
"--ens", "",
"--start", "500",
"--nconf", "1000",
]
parsed_args = parse_commandline(args)

# parsed_args = parse_commandline(ARGS)


const NFL = 1  # number of flavors, hardcoded to be 1 for now
const NL0 = parsed_args["L"]
const NT0 = parsed_args["T"]

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Nf2)

data = (
    P = zeros(Float64, NFL, NFL, NT0),
    disc = zeros(Float64, NFL, NFL, NT0),
    Delta = zeros(Float64, NFL, NT0),
    R = zeros(Float64, NFL, NFL, NFL, NFL, NT0),
    C = zeros(Float64, NFL, NFL, NFL, NFL, NT0),
    D = zeros(Float64, NFL, NFL, NFL, NFL, NT0),
    VV = zeros(Float64, NFL, NFL, NFL, NFL, NT0),
    V = zeros(Float64, NFL, NFL, NT0)
)
Data = typeof(data)

function reset_data(data::Data)
    data.P .= 0.0
    data.disc .= 0.0
    data.Delta .= 0.0
    data.R .= 0.0
    data.C .= 0.0
    data.D .= 0.0
    data.VV .= 0.0
    data.V .= 0.0
    return nothing
end

"""
- Compute connected 2-point P traces and save them into data.P_tt0[ifl, jfl, t]
- Compute disconnected 1-point Delta traces separately and save them to data.Delta[ifl, t]
- Compute connected 4-point traces and save them into data.R[ifl1, ifl2, ifl3, ifl4, t] or data.C[ifl1, ifl2, ifl3, ifl4, t]
- Compute disconnected 4-point traces with 2 P traces and save them into data.D[ifl1, ifl2, ifl3, ifl4, t]
- Compute connected 2-point at time t traces and save them into data.V[ifl, jfl, t]
"""
function correlators(data::Data, corrws::U1exCorrelator, u1ws)
    reset_data(data)

    for ifl in 1:NFL
        ex_disconnected_correlator(corrws, u1ws, ifl)
        data.Delta[ifl,:] .+= corrws.result
        for jfl in 1:NFL
            for t0 in 1:NT0
                ex_connected_correlator_tt0(corrws, u1ws, t0, ifl, jfl)
                for t in 1:NT0
                    tt=((t-t0+NT0)%NT0+1);
                    data.P[ifl,jfl,tt] += corrws.result[t] / NT0
                end
            end
        end
    end
    compute_disconnected!(data)

    for ifl1 in 1:NFL
        for ifl2 in 1:NFL
            traces = []
            ex_connected_correlator_tt(corrws, u1ws, ifl1, ifl2)
            data.V[ifl1,ifl2,:] .+= corrws.result
            for t0 in 1:NT0
                ex_connected_correlator_tt0(corrws, u1ws, t0, ifl1, ifl2)
                push!(traces, corrws.result)
            end
            for ifl3 in 1:NFL
                for ifl4 in 1:NFL
                    for t0 in 1:NT0
                        ex_connected_correlator_tt0(corrws, u1ws, t0, ifl3, ifl4)
                        for t in 1:NT0
                            tt=((t-t0+NT0)%NT0+1);
                            data.D[ifl1,ifl2,ifl3,ifl4,tt] += corrws.result[t] * traces[t0][t] / NT0
                        end
                        ex_4point_connected_correlator_ttt0t0(corrws, u1ws, t0, ifl1, ifl2, ifl3, ifl4)
                        for t in 1:NT0
                            tt=((t-t0+NT0)%NT0+1);
                            data.R[ifl1,ifl2,ifl3,ifl4,tt] += corrws.result[t] / NT0
                        end
                        ex_4point_connected_correlator_tt0tt0(corrws, u1ws, t0, ifl1, ifl2, ifl3, ifl4)
                        for t in 1:NT0
                            tt=((t-t0+NT0)%NT0+1);
                            data.C[ifl1,ifl2,ifl3,ifl4,tt] += corrws.result[t] / NT0
                        end
                    end
                end
            end
        end
    end
    compute_disconnected4!(data)

    return nothing
end

"""
Compute all combinations of disconnected traces and save them to data.disc[ifl, jfl, t]
"""
function compute_disconnected!(data::Data)
    data.disc .= 0.0
    for ifl in 1:NFL, jfl in ifl:NFL
        for t in 1:NT0, tt in 1:NT0
            data.disc[ifl, jfl, t] += data.Delta[ifl, tt] * data.Delta[jfl, (tt+t-1-1)%NT0+1] / NT0
        end
    end
    return nothing
end

"""
Compute all combinations of disconnected traces and save them to data.VV[ifl1, ifl2, ifl3, ifl4, t]
"""
function compute_disconnected4!(data::Data)
    data.VV .= 0.0
    for ifl1 in 1:NFL, ifl2 in ifl1:NFL
        for ifl3 in 1:NFL, ifl4 in ifl3:NFL
            for t in 1:NT0, tt in 1:NT0
                data.VV[ifl1, ifl2, ifl3, ifl4, t] += data.V[ifl1, ifl2, tt] * data.V[ifl3, ifl4, (tt+t-1-1)%NT0+1] / NT0
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

    for ifl1 in 1:NFL, ifl2 in ifl1:NFL
        Vfile = joinpath(dirpath, "measurements/exV-$ifl1$(ifl2)_confs$start-$finish.txt")
        write_vector(data.V[ifl1, ifl2, :],Vfile)
        for ifl3 in 1:NFL, ifl4 in ifl3:NFL
            Rfile = joinpath(dirpath, "measurements/exR-$ifl1$ifl2$ifl3$(ifl4)_confs$start-$finish.txt")
            Cfile = joinpath(dirpath, "measurements/exC-$ifl1$ifl2$ifl3$(ifl4)_confs$start-$finish.txt")
            Dfile = joinpath(dirpath, "measurements/exD-$ifl1$ifl2$ifl3$(ifl4)_confs$start-$finish.txt")
            VVfile = joinpath(dirpath, "measurements/exVV-$ifl1$ifl2$ifl3$(ifl4)_confs$start-$finish.txt")
            write_vector(data.R[ifl1, ifl2, ifl3, ifl4, :],Rfile)
            write_vector(data.C[ifl1, ifl2, ifl3, ifl4, :],Cfile)
            write_vector(data.D[ifl1, ifl2, ifl3, ifl4, :],Dfile)
            write_vector(data.VV[ifl1, ifl2, ifl3, ifl4, :],VVfile)
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
    construct_invgD!(pws, model)
    correlators(data, pws, model)
    save_data(data, dirname(cfile))
    save_topcharge(model, dirname(cfile))
end
close(fb)

