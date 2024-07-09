using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ProgressBars

# Params
beta = 5.0 # value of the coupling of the theory
lsize = 20 # lattice size
BC = PeriodicBC # Boundary Condition
tau = 1.0 # trajectory length of Moledular Dynamics evolution
nsteps = 10 # number of steps of integration, tune to have 80% or more

# Store configurations
nsamples = 5
filename = "test"

# Create model workspace U1 quenched theory
model = U1Quenched(
    Float64,
    beta = beta,
    iL = (lsize, lsize),
    BC = BC,
)

# Run HMC sampling
LFTU1.coldstart!(model)

# Create sampler workspace
samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(tau, nsteps)))

# Sample 50,000 configurations and save them to "test.bdio"
for _ in ProgressBar(1:nsamples)
    sample!(model, samplerws)
    save_cnfg(joinpath(@__DIR__, filename*".bdio"), model)
end
