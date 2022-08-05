using NNlib: relu

function initialize(din,dout)
	scale = √(6/(din+dout)) |> Float32  # Glorot uniform dist
	A = scale * (2rand(Float32,din,dout).-1)
end

abstract type AbstractNetworkLayer{T<:Real} end
struct Layer{T<:Real} <: AbstractNetworkLayer{T}
	W::Matrix{T}
	b::Vector{T}
	activation::Bool
end
Layer((din,dout)::Pair, activation::Bool = true) = Layer(initialize(din,dout), zeros(Float32, dout), activation)
function (l::Layer{T})(x::Vector{T}) where T
	y = muladd(l.W', x, l.b)
	ifelse(l.activation, relu.(y), y)
end

struct ResidualLayer{T<:Real} <: AbstractNetworkLayer{T}
	W::Matrix{T}
	b::Vector{T}
	eta::T
	activation::Bool
end
ResidualLayer((din,dout)::Pair, eta::T = 1f0, activation::Bool = true) where T = ResidualLayer{T}(initialize(din,dout), zeros(T, dout), eta, activation)
function (l::ResidualLayer{T})(x::Vector{T}) where T
	y = muladd(l.W', x, l.b)
	y = ifelse(l.activation, relu.(y), y)
	@. y + l.eta * x
end

function (ls::Vector{<:AbstractNetworkLayer{T}})(x::Vector{T}) where T
	for l in ls
		x = l(x)
	end
	return x
end

# SSE loss
C(ŷ, y) = 0.5sum(abs2, ŷ .- y)
# X-entropy loss (+softmax)
function X(ŷ, y)
	# normalize
	n = maximum(abs.(ŷ))
	# log softmax
	e = exp2.(ŷ / n)
	e ./= sum(e)
	ls = log.(e)
	return -sum(ls .* y)
end
