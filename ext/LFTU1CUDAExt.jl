module LFTU1CUDAExt

using LFTU1
import CUDA

LFTU1.to_device(::Union{CUDA.CUDABackend,CUDA.CuDevice}, x) = CUDA.CuArray(x) 
LFTU1.allowscalar(::CUDA.CUDABackend) = CUDA.allowscalar(true)
LFTU1.disallowscalar(::CUDA.CUDABackend) = CUDA.allowscalar(false)

end
