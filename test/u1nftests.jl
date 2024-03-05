using LFTSampling
using LFTU1
using LinearAlgebra

let
    beta = 5.555
    lsize = 8

    # U1 Nf Theory parameters

    ns_rat = [3, 5, 2]
    r_as = [sqrt(0.45),sqrt(0.45),sqrt(0.45)]
    r_bs = [sqrt(22),sqrt(22),sqrt(22)]
    masses = [0.6, 0.3, 0.4]

    model = U1Nf(Float64,
                 beta = beta,
                 iL = (lsize, lsize),
                 am0 = masses,
                 BC = PeriodicBC,
                 ns_rat = ns_rat,
                 r_as = r_as,
                 r_bs = r_bs,
                )

    samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))

    LFTU1.randomize!(model)
    hini = generate_pseudofermions!(model, samplerws)
    hfin = 0.0
    for j in 1:length(model.params.am0)
        LFTU1.MultiCG(samplerws.X, samplerws.F[j], model.params.am0[j], model.rprm[j], model)
        hfin += dot(samplerws.X, samplerws.F[j]) |> real
    end
    @testset "Pseudofermion generation" begin
        @test isapprox(hini, hfin)
    end


    LFTU1.coldstart!(model)

    for i in 1:10
        sample!(model, samplerws)
    end

    @testset "HMC reversibility" begin
        model_bckp = deepcopy(model)
        LFTSampling.reversibility!(model, samplerws)
        ΔU = model.U .- model_bckp.U
        @test isapprox(zero(model.PRC), mapreduce(x -> abs2(x), +, ΔU), atol = 1e-15)
    end

    @testset verbose = true "HMC force" begin
        ΔF = LFTSampling.force_test(model, samplerws, 1e-7)
        @test isapprox(zero(model.PRC), ΔF, atol = 1e-5)
    end
end
