import LinearAlgebra
using SparseArrays
include("sparse.jl")

struct LazyJac{T<:Real}# <: AbstractMatrix
    x::AbstractVector{T}
    bi::AbstractVector{Bool}
end
Base.size(J::LazyJac) = (length(J.bi), length(J.bi)*length(J.x))
# function Base.Array(J::LazyJac)
#     nout = length(J.bi)
#     X = Iterators.repeated(J.x', nout) |> collect
#     return BlockDiagonal(J.bi .* X) |> Array
# end

function Base.:*(A::LinearAlgebra.Adjoint{T, <:AbstractVector{T}}, J::LazyJac{T}) where T
    SparseColumn(J.x, A', J.bi)
end

function Base.:*(A::LinearAlgebra.Adjoint{T, <:AbstractMatrix{T}}, J::LazyJac{T}) where T
    @warn "Computing MJPs instead of VJPs uses a lot of memory and is not optimized!"
    z = zero(J.x)
    out = Array{T}(undef, size(A, 1), length(J.bi), length(J.x))
    @inbounds for i = 1:length(J.bi)
        out[i,:] = ifelse(b, dot(A[i,:], J.x), z)
    end
    return out
end
