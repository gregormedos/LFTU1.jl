import LFTSampling: sample!, generate_momenta!, Hamiltonian, update_momenta!, update_fields!

# ======================= #
# ===== General U1 ====== #
# ======================= #

KernelAbstractions.@kernel function U1generate_momenta!(mom, Nx, Ny, ::Type{BC}) where BC <: AbstractBoundaryCondition

    i1, i2 = @index(Global, NTuple)

    mom[i1, i2, 1] = randn(eltype(mom))
    mom[i1, i2, 2] = randn(eltype(mom))

    if BC == OpenBC
        if i1 == Nx
            mom[i1, i2, 1] = zero(eltype(mom))
        end
        if i2 == Ny
            mom[i1, i2, 2] = zero(eltype(mom))
        end
    end
end

function generate_momenta!(U1ws::U1, hmcws::AbstractHMC)
    lp = U1ws.params
    event = U1generate_momenta!(U1ws.device)(hmcws.mom, lp.iL[1], lp.iL[2], lp.BC, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    wait(event)
    return nothing
end

# function generate_momenta!(U1ws::U1, hmcws::AbstractHMC)
#     # Create momenta for U1
#     hmcws.mom .= to_device(U1ws.device, randn(U1ws.PRC, size(hmcws.mom)))
#     return nothing
# end

function Hamiltonian(U1ws::U1, hmcws::AbstractHMC)
    H = mapreduce(x -> x^2, +, hmcws.mom)/2.0 + action(U1ws, hmcws)
    return H
end

function update_fields!(U1ws::T, epsilon, hmcws::AbstractHMC) where T <: U1
    lp = U1ws.params
    event = U1_update_field!(U1ws.device)(U1ws.U, hmcws.mom, epsilon, ndrange=(lp.iL[1], lp.iL[2]), workgroupsize=U1ws.kprm.threads)
    wait(event)
    return nothing
end

KernelAbstractions.@kernel function U1_update_field!(U, mom, epsilon)

    i1, i2 = @index(Global, NTuple)

    for id in 1:2
        U[i1,i2,id] = complex(cos(epsilon*mom[i1,i2,id]), sin(epsilon*mom[i1,i2,id])) * U[i1,i2,id]
    end
end


# ======================= #
# ===== U1 Quenched ===== #
# ======================= #

function flip_momenta_sign!(hmcws::U1quenchedHMC)
    hmcws.mom .= .- hmcws.mom
    return nothing
end

function update_momenta!(U1ws::U1Quenched, epsilon, hmcws::AbstractHMC)
    force!(U1ws, hmcws)
    hmcws.mom .= hmcws.mom .+ epsilon .* (hmcws.frc1 .+ hmcws.frc2)
    return nothing
end


