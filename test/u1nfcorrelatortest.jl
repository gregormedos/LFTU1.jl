import LinearAlgebra: norm

let
    """
    Computes connected and disconnected (exact) traces with sink at t0 using
    point sources
    """
    function compute_trace(t0, corrws::U1Correlator, u1ws, ifl)
        S0 = corrws.S0
        S = corrws.S
        R = corrws.R
        lp = u1ws.params
        S0 .= zero(ComplexF64)
        trc = zeros(ComplexF64, 24)
        trd = zero(ComplexF64)
        for x in 1:24, s in 1:2
            S0[x,t0,s] = 1.0
            ## Solve g5D R = S0 for S for Flavor ifl
            iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0[ifl]), S0, model.sws, model)
            gamm5Dw!(R[ifl], S, model.params.am0[ifl], model)
            for t in 1:24
                trc[t] += dot(R[ifl][:,t,:], R[ifl][:,t,:]) / lp.iL[1]
            end
            trd += dot(S0, R[ifl]) / sqrt(lp.iL[1])
            S0[x,t0,s] = 0.0
        end
        return real.(trc), real(trd)
    end

    model = U1Nf(Float64,
                 beta = 4.0,
                 iL = (24, 24),
                 am0 = [0.02, 0.02],
                 BC = PeriodicBC,
                 # device = device,
                )

    N0 = model.params.iL[1]

    randomize!(model)

    smplr = HMC(integrator = OMF4(1.0, 4))
    samplerws = LFTSampling.sampler(model, smplr)

    for i in 1:10
        sample!(model, samplerws)
    end

    pws = U1exCorrelator(model)
    construct_invgD!(pws, model)

    P = zeros(Float64, length(pws.result))
    for it in 1:N0
        ex_connected_correlator(pws, model, it, 1, 2)
        for t in 1:N0
            tt=((t-it+N0)%N0+1);
            P[tt] += pws.result[t] / N0
        end
    end


    pws2 = U1Correlator(model)

    P2 = zeros(Float64, length(pws.result))
    Delta = zeros(Float64, N0)

    for it in 1:N0
        trc, trd = compute_trace(it, pws2, model, 1)
        Delta[it] += trd
        for t in 1:N0
            tt=((t-it+N0)%N0+1);
            P2[tt] += trc[t] / N0
        end
    end

    @testset verbose = true "Exact connected correlator" begin
        @test isapprox(norm(P2 .- P), 0, atol = 1e-14)
    end

    ex_disconnected_correlator(pws, model, 1)

    @testset verbose = true "Exact disconnected correlator" begin
        @test isapprox(norm(Delta .- pws.result), 0, atol = 1e-14)
    end
end

