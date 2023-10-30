using LFTU1

beta = 5.555
lsize = 8

model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = OpenBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Quenched)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Quenched OBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end


model = U1Quenched(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   BC = PeriodicBC,
                  )

LFTU1.randomize!(model)
fname = "U1iotest.bdio"
LFTU1.save_cnfg(fname, model)
fb, model2 = LFTU1.read_cnfg_info(fname, U1Quenched)
LFTU1.read_next_cnfg(fb, model2)
rm(fname, force=true)

@testset "Quenched PBC" begin
    @test model.params == model2.params
    @test model.U == model2.U
end
