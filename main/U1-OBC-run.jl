# U1
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using BDIO

beta = 5.555
lsize = 64

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))

LFTU1.randomize!(model)

fname = "run-b$beta-L$lsize-nc10000.bdio"
fb = BDIO_open(fname, "w","U1 Quenched")
BDIO_close!(fb)

# Thermalize
for i in 1:10000
    @time sample!(model, samplerws)
end

# Run
for i in 1:10000
    @time sample!(model, samplerws)
    fb = BDIO_open(fname, "a","U1 Quenched")
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
    BDIO_write!(fb,model)
    BDIO_write_hash!(fb)
    BDIO_close!(fb)
end

