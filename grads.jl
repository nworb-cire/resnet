include("lazyjacobian.jl")
using DataStructures: Deque
using BlockDiagonals
using LinearAlgebra: Diagonal, I

# Weight jacobians
function ∂∂W(ξ::Vector{T}, bi::BitVector) where T
	nout = length(bi)
	X = Iterators.repeated(ξ', nout) |> collect
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end
function ∂∂W(l::AbstractNetworkLayer{T}, x::Vector{T}) where T
	nout = length(l.b)
	X = Iterators.repeated(x', nout) |> collect
	bi = _∂∂b(l,x)
	bi, x
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end

# Bias jacobians
function _∂∂b(l::AbstractNetworkLayer{T}, x::Vector{T}) where T
	if l.activation
		bi = muladd(l.W',x,l.b) .≥ 0
	else
		bi =  ones(length(l.b)) |> BitVector
	end
	return bi
end
function ∂∂b(l::AbstractNetworkLayer{T}, x::Vector{T}) where T
	return Vector{Float32}(_∂∂b(l,x))
end
∂∂b(ξ::Vector{T}, bi::BitVector) where T = Vector{T}(bi)

# Passthrough jacobians
function ∂∂ξ(l::Layer{T}, x::Vector{T}) where T
	bi = _∂∂b(l,x)
	return bi .* l.W'
end
function ∂∂ξ(l::ResidualLayer{T}, x::Vector{T}) where T
	bi = _∂∂b(l,x)
	return l.eta*I(length(x)) .+ (bi .* l.W')
end
∂∂ξ(l::Layer{T}, bi::BitVector) where T = ∂∂ξ(l.W, bi)
∂∂ξ(W::Matrix{T}, bi::BitVector) where T = bi .* W'
∂∂ξ(l::ResidualLayer{T}, bi::BitVector) where T = ∂∂ξ(l.W, l.eta, bi)
∂∂ξ(W::Matrix{T}, η::T, bi::BitVector) where T = η*I(length(bi)) .+ bi .* W'

# Cost jacobian
#  TODO: Allow specification of loss function
∂C∂(ξ, y) = (ξ .- y)'
