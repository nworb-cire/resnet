include("types.jl")
include("lazyjacobian.jl")
using BlockDiagonals
using LinearAlgebra: Diagonal, I

# Weight jacobians
function ∂∂W(ξ::AbstractVector{T}, bi::BoolVector) where T
	nout = length(bi)
	X = Iterators.repeated(ξ', nout) |> collect
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end
function ∂∂W(l::AbstractNetworkLayer{T}, x::AbstractVector{T}) where T
	nout = length(l.b)
	X = Iterators.repeated(x', nout) |> collect
	bi = _∂∂b(l,x)
	bi, x
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end

# Bias jacobians
function _∂∂b(l::AbstractNetworkLayer{T}, x::AbstractVector{T}) where T
	if l.activation
		bi = muladd(l.W',x,l.b) .≥ 0
	else
		bi =  ones(length(l.b)) |> BitVector  # TODO: GPU
	end
	return bi
end
∂∂b(l::AbstractNetworkLayer{T}, x::AbstractVector{T}) where T = typeof(x)(_∂∂b(l,x))
∂∂b(ξ::AbstractVector{T}, bi::BoolVector) where T = typeof(ξ)(bi)

# Passthrough jacobians
∂∂ξ(l::Layer{T}, x::AbstractVector{T}) where T = _∂∂b(l,x) .* l.W'
∂∂ξ(l::ResidualLayer{T}, x::AbstractVector{T}) where T = l.eta*I(length(x)) .+ (_∂∂b(l,x) .* l.W')
∂∂ξ(l::Layer{T}, bi::BoolVector) where T = ∂∂ξ(l.W, bi)
∂∂ξ(W::Matrix{T}, bi::BoolVector) where T = bi .* W'  # TODO: LazyJac?
∂∂ξ(l::ResidualLayer{T}, bi::BoolVector) where T = ∂∂ξ(l.W, l.eta, bi)
∂∂ξ(W::Matrix{T}, η::T, bi::BoolVector) where T = η*I(length(bi)) .+ bi .* W'

# Cost jacobian
#  TODO: Allow specification of loss function
∂C∂(ξ, y) = (ξ .- y)'
