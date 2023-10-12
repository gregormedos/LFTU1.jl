using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1

using BDIO

import KernelAbstractions: CPU

beta = 5.0
lsize = 20
device = CPU()

model = U1Quenched(Float64, beta = beta, iL = (lsize, lsize), device = device)
LFTU1.randomize!(model)

fname = "test.bdio"
fb = BDIO_open(fname, "w","U1 Quenched")
BDIO_close!(fb)

fb = BDIO_open(fname, "a","U1 Quenched")
BDIO_start_record!(fb, BDIO_BIN_F64LE, 1, true)
BDIO_write!(fb,model)
BDIO_write_hash!(fb)
BDIO_close!(fb)

model2 = deepcopy(model)
LFTU1.coldstart!(model2)

fb2 = BDIO_open("test.bdio", "r")
BDIO_seek!(fb2)
BDIO_read(fb2, model2)
BDIO_close!(fb2)

