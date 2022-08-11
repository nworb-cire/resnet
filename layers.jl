using NNlib: relu
# import CUDA
# import Metal

function initialize(din,dout)
	scale = √(6/(din+dout)) |> Float32  # Glorot uniform dist
	A = scale * (2rand(Float32,din,dout).-1)
end

abstract type AbstractNetworkLayer{T<:Real} end
struct Layer{T<:Real} <: AbstractNetworkLayer{T}
	W::AbstractMatrix{T}
	b::AbstractVector{T}
	activation::Bool
end
Layer((din,dout)::Pair, activation::Bool = true) = Layer(initialize(din,dout), zeros(Float32, dout), activation)
function (l::Layer{T})(x::AbstractVector{T}) where T
	y = muladd(l.W', x, l.b)
	ifelse(l.activation, relu.(y), y)
end
# CUDA.cu(l::Layer) = Layer(cu(l.W), cu(l.b), l.activation)
if @isdefined MtlArray
	MtlArray(l::Layer) = Layer(MtlArray(l.W), MtlArray(l.b), l.activation)
end

struct ResidualLayer{T<:Real} <: AbstractNetworkLayer{T}
	W::AbstractMatrix{T}
	b::AbstractVector{T}
	eta::T
	activation::Bool
end
ResidualLayer((din,dout)::Pair, eta::T = 1f0, activation::Bool = true) where T = ResidualLayer{T}(initialize(din,dout), zeros(T, dout), eta, activation)
function (l::ResidualLayer{T})(x::AbstractVector{T}) where T
	y = muladd(l.W', x, l.b)
	y = ifelse(l.activation, relu.(y), y)
	@. y + l.eta * x
end
# CUDA.cu(l::ResidualLayer) = Layer(cu(l.W), cu(l.b), l.eta, l.activation)
if @isdefined MtlArray
	MtlArray(l::ResidualLayer) = Layer(MtlArray(l.W), MtlArray(l.b), l.eta, l.activation)
end

struct Network{T<:Real,N}
	layers::Tuple{Vararg{<:AbstractNetworkLayer{T},N}}
	in_dim
	out_dim
	hidden_dim
end
function FeedforwardNetwork(
		in_dim::Int, 
		hidden_dim::Int, 
		out_dim::Int,
		n_layers::Int,
		final_activation::Bool = false
	) where T
	n_layers < 2 && error("Need at least two layers!")
	Network((
		Layer(in_dim=>hidden_dim),
		[
			Layer(hidden_dim=>hidden_dim) 
			for _ in 1:n_layers-2
		]...,
		Layer(hidden_dim=>out_dim, final_activation)
	), in_dim, out_dim, hidden_dim)
end
Base.length(nn::Network{T,N}) where {T,N} = N

function ResidualNetwork(
		in_dim::Int, 
		hidden_dim::Int, 
		out_dim::Int,
		n_layers::Int,
		final_activation::Bool = false
	) where T
	n_layers < 2 && error("Need at least two layers!")
	Network((
		Layer(in_dim=>hidden_dim),
		[
			ResidualLayer(hidden_dim=>hidden_dim) 
			for _ in 1:n_layers-2
		]...,
		Layer(hidden_dim=>out_dim, final_activation)
	), in_dim, out_dim, hidden_dim)
end

function (nn::Network{T,N})(x::AbstractVector{T}) where {T,N}
	ŷ = nn.layers[1](x)
	for i in 2:N-1
		ŷ .= nn.layers[i](ŷ)
	end
	return nn.layers[N](ŷ)
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
