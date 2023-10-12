module LFTU1

using LFTSampling

using KernelAbstractions
import Random
import BDIO

abstract type U1 <: AbstractLFT end
abstract type U1Quenched <: U1 end
export U1Quenched

abstract type U1Parm <: LFTParm end

Base.@kwdef struct U1QuenchedParm <: U1Parm
    iL::Tuple{Int64,Int64}
    beta::Float64
end
export U1QuenchedParm

struct KernelParm
    threads::Tuple{Int64,Int64}
    blocks::Tuple{Int64,Int64}
end

KernelParm(lp::U1Parm) = KernelParm((lp.iL[1], 1), (1, lp.iL[1]))
export KernelParm


include("U1fields.jl")
export U1quenchedworkspace, coldstart!

include("U1action.jl")
export action, U1plaquette!, U1action, gauge_action, top_charge

include("U1hmc.jl")
export Hamiltonian, generate_momenta!, update_fields!, U1_update_field!, update_momenta!

include("U1io.jl")

# to_device(::CUDAKernels.CUDADevice, x) = CUDA.CuArray(x)
# to_device(::ROCKernels.ROCDevice, x) = AMDGPU.ROCArray(x)
to_device(::KernelAbstractions.CPU, x) = x

allowscalar(::KernelAbstractions.CPU) = nothing
disallowscalar(::KernelAbstractions.CPU) = nothing
# allowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(true)
# disallowscalar(::CUDAKernels.CUDADevice) = CUDA.allowscalar(false)
# allowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(true)
# disallowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(false)

# Glossary of variable name meanings

# ws = workspace
# lp = lattice parameter
# frc = force

end # module LFTU1
