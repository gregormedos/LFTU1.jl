using LFTSampling
using LFTU1
using LinearAlgebra


# U1 Nf Theory parameters

ns_rat = [3, 5, 2, 10]
r_as = [sqrt(0.45),sqrt(0.45),sqrt(0.45),sqrt(0.45)]
r_bs = [sqrt(22),sqrt(22),sqrt(22),sqrt(22)]
masses = [0.6, 0.3, 0.4, 0.1]

model = U1Nf(Float64,
                   beta = beta,
                   iL = (lsize, lsize),
                   am0 = masses,
                   BC = PeriodicBC,
                   ns_rat = ns_rat,
                   r_as = r_as,
                   r_bs = r_bs,
                  )


LFTU1.randomize!(model)

samplerws = LFTSampling.sampler(model, HMC(integrator = Leapfrog(1.0, 20)))

hini = generate_pseudofermions!(model, samplerws)

hfin = 0.0
for j in 1:length(model.params.am0)
    LFTU1.MultiCG(samplerws.X, samplerws.F[j], model.params.am0[j], model.rprm[j], model)
    global hfin += dot(samplerws.X, samplerws.F[j]) |> real
end

@testset "Pseudofermion generation" begin
    @test isapprox(hini, hfin)
end
