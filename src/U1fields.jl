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
struct U1quenchedworkspace{T, A <: AbstractArray} <: U1Quenched
    PRC::Type{T}
    U::A
    params::U1QuenchedParm
    device#::Union{KernelAbstractions.Device, ROCKernels.ROCDevice}
    kprm::KernelParm
    function U1quenchedworkspace(::Type{T}, lp::U1Parm, device, kprm) where {T <: AbstractFloat}
        U = to_device(device, ones(complex(T), lp.iL..., 2))
        return new{T, typeof(U)}(T, U, lp, device, kprm)
    end
end

function (s::Type{U1Quenched})(::Type{T}; device = CUDAKernels.CUDADevice(), kwargs...) where {T <: AbstractFloat}
    lp = U1QuenchedParm(;kwargs...)
    return U1quenchedworkspace(T, lp, device, KernelParm(lp))
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

