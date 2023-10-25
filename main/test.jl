# U1 Nf2
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using BDIO

ENV["JULIA_DEBUG"] = "all"

beta = 5.0
lsize = 20
mass = 0.2

model = U1Nf2(Float64, beta = beta, iL = (lsize, lsize), am0 = mass, BC = PeriodicBC)

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 15)))

LFTU1.randomize!(model)


@time sample!(model, samplerws)

Ss = Vector{Float64}(undef, 1000)


for i in 1:1000
    @time sample!(model, samplerws)
    Ss[i] = gauge_action(model)
end

using ADerrors

id = "test"
uws = uwreal(Ss, id)
uwp = 1 - uws/(beta*lsize^2)
uwerr(uwp)
uwp





