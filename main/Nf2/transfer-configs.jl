# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using TOML
using LFTSampling
using LFTU1
using Dates
import KernelAbstractions

# Read model parameters

configfile = "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nf2sim-b4.0-L24-m0.02_D2024-04-12-17-03-40.579/Nf2sim-b4.0-L24-m0.02_D2024-04-12-17-03-40.579.bdio"
newconfigfile = configfile*"_new"
beta = 4.0
mass = 0.02
lsize = 24
BC = PeriodicBC
device = KernelAbstractions.CPU()

model = U1Nf2(
              Float64, 
              beta = beta, 
              am0 = mass, 
              iL = (lsize, lsize), 
              BC = PeriodicBC,
              device = device,
             )

randomize!(model)

ncfgs = LFTSampling.count_configs(configfile)
fb, model = read_cnfg_info(configfile, U1Nf2)

for i in 1:ncfgs
    LFTSampling.read_next_cnfg(fb, model)
    save_cnfg(newconfigfile, model)
end
close(fb)
