module LFTU1AMD

import AMDGPU, ROCKernels

to_device(::ROCKernels.ROCDevice, x) = AMDGPU.ROCArray(x)

allowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(true)
disallowscalar(::ROCKernels.ROCDevice) = AMDGPU.allowscalar(false)

end


