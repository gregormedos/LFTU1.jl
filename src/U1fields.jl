import LFTSampling: copy!, sampler

# ======================= #
# ===== General U1 ====== #
# ======================= #

function copy!(U1ws_dst::U1, U1ws_src::U1)
    U1ws_dst.U .= U1ws_src.U
    return nothing
end

function randomize!(U1ws::U1)
    U1ws.U .= to_device(U1ws.device, exp.(im * Random.rand(U1ws.PRC, size(U1ws.U)) * 2 * pi))
    return nothing
end

function coldstart!(U1ws::U1)
    U1ws.U .= one(complex(U1ws.PRC))
    return nothing
end

# ======================= #
# ===== U1 Quenched ===== #
# ======================= #

@doc raw"""
    struct U1workspace{T}

Allocates all the necessary fields for a HMC simulation of a U(1) model:

- `PRC`: precision; must be `ComplexF64`.
- `U`: ``U`` gauge field.
- `frc`
"""
struct U1quenchedworkspace{T1, A <: AbstractArray} <: U1Quenched
    PRC::Type{T1}
    U::A
    params::U1QuenchedParm
    device#::Union{KernelAbstractions.Device, ROCKernels.ROCDevice}
    kprm::KernelParm
end

function (s::Type{U1Quenched})(::Type{T1}, ::Type{T2} = complex(T1); custom_init = nothing, device = KernelAbstractions.CPU(), kwargs...) where {T1, T2}
    lp = U1QuenchedParm(;kwargs...)
    return U1quenchedworkspace(T1, T2, lp, device, KernelParm(lp); custom_init = custom_init)
end

function U1quenchedworkspace(::Type{T1}, ::Type{T2}, lp::U1Parm, device, kprm; custom_init = nothing) where {T1,T2}
    if custom_init == nothing
        U = to_device(device, ones(T2, lp.iL..., 2))
    else
        U = custom_init
    end
    return U1quenchedworkspace{T1, typeof(U)}(T1, U, lp, device, kprm)
end

struct U1quenchedHMC{A <: AbstractArray} <: AbstractHMC
    params::HMC
    frc1::A
    frc2::A
    mom::A
end


function U1quenchedHMC(u1ws::U1Quenched, hmcp::HMCParams)
    frc1 = to_device(u1ws.device, zeros(u1ws.PRC, u1ws.params.iL..., 2))
    frc2 = similar(frc1)
    mom = similar(frc1)
    return U1quenchedHMC(hmcp, frc1, frc2, mom)
end

sampler(lftws::U1Quenched, hmcp::HMCParams) = U1quenchedHMC(lftws, hmcp)

