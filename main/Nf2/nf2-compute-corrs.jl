# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ProgressBars

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")
cfile = ARGS[1]

ncfgs = LFTSampling.count_configs(cfile)
fb, model = read_cnfg_info(cfile, U1Nf2)
pws = U1PionCorrelator(model, wdir=dirname(cfile))
pcac = U1PCACCorrelator(model, wdir=dirname(cfile))
for i in ProgressBar(1:ncfgs)
    read_next_cnfg(fb, model)
    pws(model)
    pcac(model)
    write(pws)
    write(pcac)
end
close(fb)
