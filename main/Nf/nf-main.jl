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
    infile = "main/Nf/infile.in"
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

function create_simulation_directory(wdir::String, u1ws::U1Nf)
    dt = Dates.now()
    wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
    fname = "Nfsim-b$(u1ws.params.beta)-L$(model.params.iL[1])-m$(filter(!isspace,string(u1ws.params.am0)))"*wdir_sufix
    fdir = joinpath(wdir, fname)
    configfile = joinpath(fdir, fname*".bdio")
    mkpath(fdir)
    cp(infile, joinpath(fdir,splitpath(infile)[end]))
    return configfile
end

# Read model parameters

beta = pdata["Model params"]["beta"]
masses = pdata["Model params"]["masses"]
lsize = pdata["Model params"]["L"]
BC = eval(Meta.parse(pdata["Model params"]["BC"]))

# Read HMC parameters

tau = pdata["HMC params"]["tau"]
nsteps = pdata["HMC params"]["nsteps"]
ntherm = pdata["HMC params"]["ntherm"]
ntraj = pdata["HMC params"]["ntraj"]
discard = pdata["HMC params"]["discard"]
integrator = eval(Meta.parse(pdata["HMC params"]["integrator"]))

# Read HMC parameters

ns_rat = pdata["RHMC params"]["nrhmc"]
r_as = pdata["RHMC params"]["r_as"]
r_bs = pdata["RHMC params"]["r_bs"]


# Working directory

wdir = pdata["Working directory"]["wdir"]
cntinue = pdata["Working directory"]["continue"]
cntfile = pdata["Working directory"]["cntfile"]


model = U1Nf(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   am0 = masses,
                   BC = PeriodicBC,
                   device = device,
                  )

randomize!(model)
smplr = HMC(integrator = integrator(tau, nsteps+1))
samplerws = LFTSampling.sampler(model, smplr)

@info "Creating simulation directory"

if cntinue == true
    @info "Reading from old simulation"
    configfile = cntfile
    ncfgs = LFTSampling.count_configs(configfile)
    fb, model = read_cnfg_info(configfile, U1Nf)
    LFTSampling.read_cnfg_n(fb, ncfgs, model)
    close(fb)
else
    @info "Creating simulation directory"
    ncfgs = 0
    configfile = create_simulation_directory(wdir, model)
end

model.rprm .= LFTU1.get_rhmc_params(ns_rat, r_as, r_bs)

logio = open(dirname(configfile)*"/log.txt", "a+")
Wio = open(dirname(configfile)*"/reweighting_factor.txt", "a+")
logger = SimpleLogger(logio)
global_logger(logger)

@info "U(1) NF SIMULATION" model.params smplr

reasonable_bound = 2 * lsize^2 * model.rprm[1].delta^2
if  reasonable_bound > 10^(-4) # reasonable if <= 10^(-4)
    @info "2Vδ² = $(reasonable_bound) > 10⁻⁴"
end

@info "Starting thermalization"

function log_spectral_range(u1ws::U1Nf)
    for j in 1:length(u1ws.params.am0)
        lambda_min, lambda_max =  power_method(u1ws, u1ws.params.am0[j], iter=10000)
        @info "m$(j): r_a = $(real(sqrt(lambda_min))), r_b = $(real(sqrt(lambda_max)))"
        if real(sqrt(lambda_min)) < r_as[j] || real(sqrt(lambda_max)) > r_bs[j]
            @warn "OUT OF SPECTRAL RANGE"
        end
    end
end

if cntinue == true
    @info "Skipping thermalization"
else
    @info "Starting thermalization"

    @info "10 FIRST THERMALIZATION STEPS WITH nsteps + 1 INTEGRATION STEPS"
    for i in 1:10
        @info "THERM STEP $i"
        @time sample!(model, samplerws)
    end

    @info "REMAINING THERMALIZATION WITH nsteps INTEGRATION STEPS"
    smplr = HMC(integrator = integrator(tau, nsteps))
    samplerws = LFTSampling.sampler(model, smplr)

    for i in 11:ntherm
        @info "THERM STEP $i"
        @time sample!(model, samplerws)
        W_N = LFTU1.reweighting_factor(model)
        @info "W_N = $(W_N)"
        if i%10 == 0
            log_spectral_range(model)
        end
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
    W_N = LFTU1.reweighting_factor(model)
    @info "W_N = $(W_N)"
    write(Wio, "$(W_N)\n")
    save_cnfg(configfile, model)
    if i%100 == 0
        log_spectral_range(model)
    end
    flush(logio)
    flush(Wio)
end

@info "Simulation finished succesfully"
close(logio)
close(Wio)
