using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ArgParse

parse_commandline() = parse_commandline(ARGS)
function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
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
"--ens", "",
"--start", "1",
"--nconf", "1000",
]
parsed_args = parse_commandline(args)

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

function save_observable(model, dirpath)
    qfile = joinpath(dirpath,"action$start-$finish.txt")
    global io_stat = open(qfile, "a")
    write(io_stat, "$(action(model))\n")
    close(io_stat)
end

fb, model = read_cnfg_info(cfile, U1Nf2)
for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    save_action(model, dirname(cfile))
end
close(fb)
