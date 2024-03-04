using LinearAlgebra


# U1 Theory parameters
lsize = 8
beta = 5.0
mass = 0.6

model = LFTU1.U1Nf2(Float64, iL = (lsize, lsize), beta = beta, am0 = mass, BC = PeriodicBC)

alg = HMC(
          integrator = Leapfrog(1.0, 10),
         )

hmcws = LFTU1.sampler(model, alg)

LFTU1.randomize!(model)
LFTU1.generate_pseudofermions!(model, hmcws)

@testset "Pseudofermion generation" begin
    xi = similar(hmcws.X)
    invert!(xi, gamm5Dw_sqr_msq!, hmcws.F, model.sws, model)
    @test isapprox(dot(xi, hmcws.F), dot(hmcws.X, hmcws.X))
end

@testset "HMC reversibility" begin
    model_bckp = deepcopy(model)
    LFTSampling.reversibility!(model, hmcws)
    ΔU = model.U .- model_bckp.U
    @test isapprox(zero(model.PRC), mapreduce(x -> abs2(x), +, ΔU), atol = 1e-15)
end

@testset verbose = true "HMC force" begin
    ΔF = LFTSampling.force_test(model, hmcws, 1e-7)
    @test isapprox(zero(model.PRC), ΔF, atol = 1e-5)
end
