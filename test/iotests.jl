using LFTU1

let
    beta = 5.555
    lsize = 8

    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = OpenBC,
                      )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Quenched)
    @testset "Quenched OBC" begin
        for i in 1:length(ens)
            @test ens[i].params == rens[i].params
            @test ens[i].U == rens[i].U
        end
    end
    rm(fname, force=true)



    model = U1Quenched(Float64,
                       beta = beta,
                       iL = (lsize, lsize),
                       BC = PeriodicBC,
                      )
    LFTU1.randomize!(model)
    fname = "qriotest.bdio"
    isfile(fname) && error("File already exists!")
    ens = [deepcopy(model) for i in 1:10]
    for i in 1:length(ens)
        LFTU1.randomize!(ens[i])
        save_cnfg(fname, ens[i])
    end
    rens = LFTSampling.read_ensemble(fname, U1Quenched)
    @testset "Quenched PBC" begin
        for i in 1:length(ens)
            @test ens[i].params == rens[i].params
            @test ens[i].U == rens[i].U
        end
    end
    rm(fname, force=true)


    # mass = 0.6
    # model = U1Nf2(Float64,
    #               beta = beta,
    #               am0 = mass,
    #               iL = (lsize, lsize),
    #               BC = PeriodicBC,
    #              )

    # LFTU1.randomize!(model)
    # fname = "U1iotest.bdio"
    # LFTU1.save_cnfg(fname, model)
    # fb, model2 = LFTU1.read_cnfg_info(fname, U1Quenched)
    # LFTU1.read_next_cnfg(fb, model2)
    # rm(fname, force=true)

    # @testset "Nf=2 PBC" begin
    #     @test model.params == model2.params
    #     @test model.U == model2.U
    # end

end
