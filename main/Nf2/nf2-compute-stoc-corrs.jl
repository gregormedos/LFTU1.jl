# Quantum Rotor
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

        "--nsrc"
        help = "number of sources"
        required = false
        arg_type = Int
        default = 2

        "--ens"
        help = "path to ensemble with configurations"
        required = true
        arg_type = String
        # default = "configs/"
    end
    return parse_args(args, s)
end

# args = [
# "-L", "24",
# "--ens", "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nfsim-b4.0-L24-m[0.02, 0.02]_D2024-04-18-18-42-12.272/Nfsim-b4.0-L24-m[0.02, 0.02]_D2024-04-18-18-42-12.272.bdio",
# "--start", "2",
# "--nconf", "10",
# "--nsrc", "2",
# ]
# parsed_args = parse_commandline(args)

parsed_args = parse_commandline(ARGS)



const NFL = 2  # number of flavors, hardcoded to be 2 by now
const N0 = parsed_args["L"]
const NSRC = parsed_args["nsrc"]

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Nf2)


function reset_data(data)
    data.P .= 0.0
    data.Delta .= 0.0
    data.disc .= 0.0
    return nothing
end

function save_data(data, dirpath)
    for ifl in 1:2, jfl in ifl:2
        connfile = joinpath(dirpath,"measurements/conn-$ifl$(jfl)_confs$start-$finish.txt")
        discfile = joinpath(dirpath, "measurements/disc-$ifl$(jfl)_confs$start-$finish.txt")
        write_vector(data.P[ifl, jfl, :],connfile)
        write_vector(data.disc[ifl, jfl, :],discfile)
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


"""
- Compute connected traces and save them into data.P[ifl, jfl, t], already
  averaging over the number of sources. 
- Compute disconnected traces separately and save them to data.Delta[ifl, isrc, t]
""" 
function correlators(data, corrws, u1ws, nsrc)
    reset_data(data)
    for isrc in ProgressBar(1:nsrc)
        for it in 1:N0
            random_source(it,corrws,u1ws)
            for ifl in 1:2
                disconnected_correlator(corrws, u1ws, ifl)
                data.Delta[ifl,isrc,:] .+= corrws.result
                for jfl in ifl:2
                    connected_correlator(corrws, u1ws, ifl, jfl)
                    for t in 1:N0
                        tt=((t-it+N0)%N0+1);
                        data.P[ifl, jfl, tt] += corrws.result[t] ./ N0 ./ nsrc
                    end
                end
            end
        end
    end
end

function compute_disconnected!(data, nsrc)
    data.disc .= 0.0
    for ifl in 1:2, jfl in ifl:2
        for isrc in 1:nsrc, jsrc in 1:nsrc
            if jsrc != isrc
                for t in 1:N0, tt in 1:N0
                    data.disc[ifl, jfl, t] += data.Delta[ifl, isrc, tt] * data.Delta[jfl, jsrc, (tt+t-1-1)%N0+1] / N0 / nsrc / (nsrc - 1)
                end
            end
        end
    end
    return nothing
end


data = (
    nc = 0,
    P = zeros(Float64, NFL, NFL, N0),
    disc = zeros(Float64, NFL, NFL, N0),
    Delta = zeros(Float64, NFL, NSRC, N0)
)

pws = U1Correlator(model, wdir=dirname(cfile))
for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    correlators(data, pws, model, NSRC)
    compute_disconnected!(data, NSRC)
    save_data(data, dirname(cfile))
    save_topcharge(model, dirname(cfile))
end
close(fb)
