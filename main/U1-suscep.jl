# U1
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1

import KernelAbstractions: CPU

import CUDAKernels: CUDADevice

beta = 5.555
lsize = 64
# device = CPU()
# device = CUDADevice()

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                   # device = device
                  )

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))

LFTU1.randomize!(model)

@time sample!(model, samplerws)

top_charge(model)

Qs = Vector{Float64}()

Ss = Vector{Float64}()

for i in 1:10000
    # print(i)
    sample!(model, samplerws)
    # Q = top_charge(model)
    S = action(model)
    push!(Ss, S)
end


using ADerrors

histogram(Qs)

ID = "test"

uwQ2s = uwreal(Qs.^2, ID)

uwerr(uwQ2s)

uwQ2s

uwS = uwreal(Ss, ID)

uwP = 1 - uwS/(model.params.beta*(model.params.iL[1]-1)^2)
uwerr(uwP)
uwP

uwW = -log(uwP)
uwerr(uwW)
uwW

"0.89338(7)"
