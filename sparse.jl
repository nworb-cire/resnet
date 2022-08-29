using LoopVectorization

struct SparseColumn{T<:Real}
    x::AbstractVector{T}
    a::AbstractVector{T}
    bi::AbstractVector{Bool}
end
Base.size(J::SparseColumn) = (length(J.x), length(J.a))
Base.:-(J::SparseColumn) = SparseColumn(-J.x, J.a, J.bi)
Base.:/(J::SparseColumn, s) where T = SparseColumn(J.x, J.a / s, J.bi)

addto!(A::AbstractMatrix{T}, J::SparseColumn{T}) where T = addto!(A, J, oneunit(T))
function addto!(A::AbstractMatrix{T}, J::SparseColumn{T}, c::T) where T
    @inbounds @fastmath A .+= c*J.x*(J.bi.*J.a)'
end
addto!(A, ::Nothing, ::Any) = A
Base.Array(J::SparseColumn{T}) where T = addto!(zeros(T, size(J)), J)

function Base.:+(A::AbstractMatrix{T}, J::SparseColumn{T}) where T
    @warn "Please use the inplace method `addto!` instead."
    B = copy(A)
    addto!(B, J)
    B
end

struct SparseColumns{T<:Real}
    cols::Dict{Int,<:AbstractVector{T}}
    width::Int
end
function SparseColumns{T}(J::SparseColumn{T})::SparseColumns{T} where T
    cols = Dict{Int,typeof(J.x)}()
    for i ∈ (first(p) for p in pairs(J.bi) if last(p))
        cols[i] = J.a[i] * J.x
    end
    w = length(J.bi)
    SparseColumns{T}(cols, w)
end
addto!(::Nothing, K::SparseColumn{T}) where T = SparseColumns{T}(K)
function addto!(J::SparseColumns{T}, K::SparseColumn{T}, c::T)::SparseColumns{T} where T
    @assert J.width == length(K.bi)
    ks = keys(J.cols)
    for i ∈ (first(p) for p in pairs(K.bi) if last(p))
        val = c*K.x*K.a[i]
        if i ∈ ks
            J.cols[i] .+= val
        else
            J.cols[i] = val
        end
    end
    J
end
