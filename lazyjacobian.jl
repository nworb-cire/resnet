import LinearAlgebra

struct LazyJac{T<:Real}# <: AbstractMatrix
    x::Vector{T}
    bi::BitVector
end
Base.size(J::LazyJac) = (length(J.bi), length(J.bi)*length(J.x))
# function Base.Array(J::LazyJac)
#     nout = length(J.bi)
#     X = Iterators.repeated(J.x', nout) |> collect
#     return BlockDiagonal(J.bi .* X) |> Array
# end

function Base.:*(A::LinearAlgebra.Adjoint{T, Vector{T}}, J::LazyJac{T}) where T
    z = zero(J.x)
    out = Matrix{T}(undef, (length(J.bi), length(J.x)))
    @inbounds for i = 1:length(J.bi)
        out[i,:] .= ifelse(J.bi[i], A[i]*J.x, z)
    end
    return out
end

function Base.:*(A::LinearAlgebra.Adjoint{T, Matrix{T}}, J::LazyJac{T}) where T
    z = zero(J.x)
    out = Array{T}(undef, size(A, 1), length(J.bi), length(J.x))
    for i = 1:length(J.bi)
        out[i,:] = ifelse(b, dot(A[i,:], J.x), z)
    end
    return out
end
