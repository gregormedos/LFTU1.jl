# U1
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1

import KernelAbstractions: CPU

beta = 5.0
lsize = 20
device = CPU()

model = U1Quenched(Float64, beta = beta, iL = (lsize, lsize), device = device)

sampler = HMC(
              integrator = Leapfrog(1.0, 10),
             )

samplerws = LFTSampling.sampler(model, sampler)

LFTU1.randomize!(model)

@time sample!(model, samplerws)
top_charge(model)

Qs = Vector{Float64}()

Ss = Vector{Float64}()

for i in 1:100000
    # print(i)
    sample!(model, samplerws)
    # Q = top_charge(model)
    S = action(model)
    push!(Ss, S)
end


using ADerrors, Plots

histogram(Qs)

ID = "test3"

uwQ2s = uwreal(Qs.^2, ID)

uwerr(uwQ2s)

uwQ2s

uwS = uwreal(Ss, ID)

uwP = 1 - uwS/(model.params.beta*model.params.iL[1]^2)
uwerr(uwP)
uwP

0.89338(7)
