# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using TOML

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")

if length(ARGS) == 1
    infile = ARGS[1]
else
    infile = "main/quenched/infile.in"
end
pdata = TOML.parsefile(infile)
device = pdata["Model params"]["device"]
if device == "CUDA"
    import CUDA
    device = CUDA.device()
elseif device == "CPU"
    import KernelAbstractions
    device = KernelAbstractions.CPU()
else
    error("Only acceptable devices are CUDA or CPU")
end

using LFTSampling
using LFTU1
using Dates
using Logging

function create_simulation_directory(wdir::String, u1ws::U1Quenched)
    dt = Dates.now()
    wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
    fname = "quenchedsim-b$(u1ws.params.beta)-L$(model.params.iL[1])-T$(model.params.iL[2])"*wdir_sufix
    fdir = joinpath(wdir, fname)
    configfile = joinpath(fdir, fname*".bdio")
    mkpath(fdir)
    cp(infile, joinpath(fdir,splitpath(infile)[end]))
    return configfile
end

# Read model parameters

beta = pdata["Model params"]["beta"]
lsize = pdata["Model params"]["L"]
tsize = pdata["Model params"]["T"]
BC = eval(Meta.parse(pdata["Model params"]["BC"]))

# Read HMC parameters

tau = pdata["HMC params"]["tau"]
nsteps = pdata["HMC params"]["nsteps"]
ntherm = pdata["HMC params"]["ntherm"]
ntraj = pdata["HMC params"]["ntraj"]
discard = pdata["HMC params"]["discard"]
integrator = eval(Meta.parse(pdata["HMC params"]["integrator"]))

# Working directory

wdir = pdata["Working directory"]["wdir"]
cntinue = pdata["Working directory"]["continue"]
cntfile = pdata["Working directory"]["cntfile"]


model = U1Quenched(
              Float64, 
              beta = beta, 
              iL = (lsize, tsize), 
              BC = PeriodicBC,
              device = device,
             )

randomize!(model)
smplr = HMC(integrator = integrator(tau, nsteps))
samplerws = LFTSampling.sampler(model, smplr)

if cntinue == true
    @info "Reading from old simulation"
    configfile = cntfile
    ncfgs = LFTSampling.count_configs(configfile)
    fb, model = read_cnfg_info(configfile, U1Quenched)
    LFTSampling.read_cnfg_n(fb, ncfgs, model)
    close(fb)
else
    @info "Creating simulation directory"
    ncfgs = 0
    configfile = create_simulation_directory(wdir, model)
end

logio = open(dirname(configfile)*"/log.txt", "a+")
logger = SimpleLogger(logio)
global_logger(logger)

@info "U(1) NF=2 SIMULATION" model.params smplr

if cntinue == true
    @info "Skipping thermalization"
else
    @info "Starting thermalization"
    for i in 1:ntherm
        @info "THERM STEP $i"
        @time sample!(model, samplerws)
        flush(logio)
    end
end

if cntinue == true
    @info "Restarting simulation from trajectory $ncfgs"
else
    @info "Starting simulation"
end

@time for i in (ncfgs+1):(ncfgs+ntraj)
    @info "TRAJECTORY $i"
    for j in 1:discard
        @time sample!(model, samplerws)
    end
    @time sample!(model, samplerws)
    save_cnfg(configfile, model)
    flush(logio)
end

@info "Simulation finished succesfully"
flush(logio)
close(logio)
