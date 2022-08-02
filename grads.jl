include("lazyjacobian.jl")
using DataStructures: Deque

# Weight jacobians
function ∂∂W(ξ::Vector{T}, bi::BitVector) where T
	nout = length(bi)
	X = Iterators.repeated(ξ', nout) |> collect
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end
function ∂∂W(l::Layer{T}, x::Vector{T}) where T
	nout = length(l.b)
	X = Iterators.repeated(x', nout) |> collect
	bi = _∂∂b(l,x)
	bi, x
    @warn "This operation is very slow and memory intensive! Please use lazy jacobians."
	return BlockDiagonal(bi .* X) |> Array
end

# Bias jacobians
function _∂∂b(l::Layer{T}, x::Vector{T}) where T
	if l.activation
		bi = muladd(l.W',x,l.b) .≥ 0
	else
		bi =  ones(length(l.b)) |> BitVector
	end
	return bi
end
function ∂∂b(l::Layer{T}, x::Vector{T}) where T
	return Vector{Float32}(_∂∂b(l,x))
end
∂∂b(ξ::Vector{T}, bi::BitVector) where T = Vector{T}(bi)

# Passthrough jacobians
function ∂∂ξ(l::Layer{T}, x::Vector{T}) where T
	bi = _∂∂b(l,x)
	return bi .* l.W'
end
∂∂ξ(l::Layer{T}, bi::BitVector) where T = ∂∂ξ(l.W, bi)
∂∂ξ(W::Matrix{T}, bi::BitVector) where T = bi .* W'

# Cost jacobian
#  TODO: Allow specification of loss function
∂C∂(ξ, y) = (ξ .- y)'


#=========================================#

DT{T} = Deque{@NamedTuple{
	ξ::Vector{T},
	bi::BitVector,
	W::Base.RefValue{Matrix{S}}
}} where {T,S}

function forward!(s::DT{T,S}, ls::Vector{Layer{T}}, x::Vector{T}, y::Vector{T}) where {S,T}
    empty!(s)
	ŷ = copy(x)
	for l in ls
		ỹ = muladd(l.W',ŷ,l.b)
		bi = ifelse(l.activation, ỹ .> 0, ones(length(ỹ)) |> BitVector)
		push!(s, (ξ=ŷ,bi=bi,W=Ref(l.W)))
		ŷ = ifelse(l.activation, relu.(ỹ), ỹ)
	end
	C(ŷ, y)
end

function grads(s::DT{T,S}, layers::Vector{Layer{T}}, x::Vector{T}, y::Vector{T}) where {S,T}
    isempty(s) && error("Stack is empty; did you run the forward pass?")
	J = ∂C∂(layers(x), y)
	∇W = Matrix{T}[]
	∇b = Vector{T}[]
	while !isempty(s)
		ξ, bi, refW = pop!(s)
		W = refW[]
		# TODO: ∂∂W is low-rank. Compute efficiently.
		∇Wₗ = reshape((J*LazyJac(ξ, bi))', size(W))
		# TODO: ∂∂b is sparse. Compute efficiently.
		∇bₗ = J'.*∂∂b(ξ, bi)
		push!(∇W,∇Wₗ)
		push!(∇b,∇bₗ)
		if !isempty(s)
			J = J*∂∂ξ(W, bi)  # Skip jacobian computation on last layer
		end
	end
	return zip(∇W, ∇b)
end
reverse! = grads
