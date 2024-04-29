# Quantum Rotor
using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using ProgressBars
using Dates
import LinearAlgebra.dot

const MAXNFL = 2  # Example size, adjust as necessary
const N0 = 24       # Example size, adjust as necessary
const MAXNSRC = 10  # Example size, adjust as necessary
nsrc = 10


length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")
cfile = ARGS[1]

ncfgs = LFTSampling.count_configs(cfile)
fb, model = read_cnfg_info(cfile, U1Nf)

struct U1Correlator <: LFTU1.AbstractU1Correlator
    name::String
    ID::String
    filepath::String
    R
    S
    S0
    result::Vector{Float64} # correlator
    history::Vector{Vector{Float64}}
    function U1Correlator(u1ws::LFTU1.U1; wdir::String = "./trash/", 
                               name::String = "U(1) correlator", 
                               ID::String = "corr_pion", 
                               mesdir::String = "measurements/", 
                               extension::String = ".txt")
        dt = Dates.now()
        wdir_sufix = "_D"*Dates.format(dt, "yyyy-mm-dd-HH-MM-SS.ss")
        lp = u1ws.params
        filepath = joinpath(wdir, mesdir, ID*wdir_sufix*extension)
        R1 = LFTU1.to_device(u1ws.device, zeros(complex(Float64), lp.iL..., 2))
        R2 = copy(R1)
        S = copy(R1)
        S0 = copy(R1)
        C = zeros(Float64, lp.iL[1])
        history = []
        mkpath(dirname(filepath))
        return new(name, ID, filepath, [R1, R2], S, S0, C, history)
    end
end


"""
Generate complex random normal source at time slice t0 storing it to corrws.S0, and solve γ₅D corrws.R[ifl] = corrws.S0 for each flavor ifl.
"""
function random_source(t0, corrws, u1ws)
    S0 = corrws.S0
    S = corrws.S
    R = corrws.R
    lp = u1ws.params
    S0 .= zero(ComplexF64)
    S0[:,t0,:] .= randn(ComplexF64, lp.iL[1],2)
    for ifl in 1:2
        ## Solve g5D R = S0 for S for Flavor ifl
        iter = LFTU1.invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0[ifl]), S0, model.sws, model)
        gamm5Dw!(R[ifl], S, model.params.am0[ifl], model)
    end
    return nothing
end

"""
Computes dot(corrws.R[ifl], corrws.R[jfl]), i.e. (D_ifl⁻¹ η, D_jfl⁻¹ η), at time slice t.
"""
function connected_correlator(corrws::U1Correlator, t, u1ws, ifl, jfl)
    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)

    # NOTE: this should be ultraslow. It may be better to put R1 and R2 into the
    # CPU prior to calling this function. For GPU, the best one can do is to
    # reduce columns of the GPU array.
    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.R[jfl][x,t,:]
        b = corrws.R[ifl][x,t,:]
        Ct += abs(dot(b,a)) / lp.iL[1]
    end
    LFTU1.disallowscalar(u1ws.device)

    return Ct
end

"""
Computes dot(corrws.S0, corrws.R[ifl]), i.e. (η^†, D_ifl⁻¹ η), at time slice t.
"""
function disconnected_correlator(corrws, t, u1ws, ifl)
    lp = u1ws.params

    Ct = zero(ComplexF64)
    a = zero(ComplexF64)
    b = zero(ComplexF64)

    # NOTE: this should be ultraslow. It may be better to put R1 and R2 into the
    # CPU prior to calling this function. For GPU, the best one can do is to
    # reduce columns of the GPU array.
    LFTU1.allowscalar(u1ws.device)
    for x in 1:lp.iL[1]
        a = corrws.S0[x,t,:]
        b = corrws.R[ifl][x,t,:]
        Ct += real(dot(b,a)) / sqrt(lp.iL[1])
    end
    LFTU1.disallowscalar(u1ws.device)

    return Ct
end

function connected_correlator(corrws::U1Correlator, u1ws, ifl, jfl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = connected_correlator(corrws, t, u1ws, ifl, jfl) |> real
    end
end

function disconnected_correlator(corrws::U1Correlator, u1ws, ifl)
    lp = u1ws.params
    for t in 1:lp.iL[1]
        corrws.result[t] = disconnected_correlator(corrws, t, u1ws, ifl) |> real
    end
end

function reset_data(data)
    data.P .= 0.0
    data.Delta .= 0.0
    data.disc .= 0.0
    return nothing
end

function save_data(data, dirpath)
    for ifl in 1:2, jfl in ifl:2
        connfile = joinpath(dirpath,"conn-$ifl$jfl.txt")
        discfile = joinpath(dirpath, "disc-$ifl$jfl.txt")
        write_vector(data.P[ifl, jfl, :],connfile)
        write_vector(data.disc[ifl, jfl, :],discfile)
    end
end

function write_vector(vec, filepath)
    global io_stat = open(filepath, "a")
    write(io_stat, "$(vec[1])")
    for i in 2:length(vec)
        write(io_stat, ",$(vec[i])")
    end
    write(io_stat, "\n")
    close(io_stat)
    return nothing
end


"""
- Compute connected traces and save them into data.P[ifl, jfl, t], already
  averaging over the number of sources. 
- Compute disconnected traces separately and save them to data.Delta[ifl, isrc, t]
""" 
function correlators(data, corrws, u1ws, nsrc)
    reset_data(data)
    for isrc in ProgressBar(1:nsrc)
        for it in 1:N0
            random_source(it,corrws,u1ws)
            for ifl in 1:2
                disconnected_correlator(corrws, u1ws, ifl)
                data.Delta[ifl,isrc,:] .+= corrws.result
                for jfl in ifl:2
                    connected_correlator(corrws, u1ws, ifl, jfl)
                    for t in 1:N0
                        tt=((t-it+N0)%N0+1);
                        data.P[ifl, jfl, tt] += corrws.result[t] ./ N0 ./ nsrc
                    end
                end
            end
        end
    end
end

function compute_disconnected!(data, nsrc)
    data.disc .= 0.0
    for ifl in 1:2, jfl in ifl:2
        for isrc in 1:nsrc, jsrc in 1:nsrc
            if jsrc != isrc
                for t in 1:N0, tt in 1:N0
                    data.disc[ifl, jfl, t] += data.Delta[ifl, isrc, tt] * data.Delta[jfl, jsrc, (tt+t-1-1)%N0+1] / N0 / nsrc / (nsrc - 1)
                end
            end
        end
    end
    return nothing
end


data = (
    nc = 0,
    P = zeros(Float64, MAXNFL, MAXNFL, N0),
    disc = zeros(Float64, MAXNFL, MAXNFL, N0),
    Delta = zeros(Float64, MAXNFL, MAXNSRC, N0)
)


pws = U1Correlator(model, wdir=dirname(cfile))
for i in ProgressBar(1:ncfgs)
    read_next_cnfg(fb, model)
    correlators(data, pws, model, nsrc)
    compute_disconnected!(data, nsrc)
    save_data(data, dirname(cfile))
end
close(fb)
