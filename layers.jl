using BlockDiagonals
using LinearAlgebra: Diagonal
using NNlib: relu


struct Layer{T<:Real}
	W::Matrix{T}
	b::Vector{T}
	activation::Bool
end

function initialize(din,dout)
	scale = √(6/(din+dout)) |> Float32  # Glorot uniform dist
	A = scale * (2rand(Float32,din,dout).-1)
end

Layer((din,dout)::Pair, activation = true) = Layer(initialize(din,dout), zeros(Float32, dout), activation)

function (l::Layer{T})(x::Vector{T}) where T
	y = muladd(l.W', x, l.b)
	if l.activation
		return relu.(y)
	else
		return y
	end
end

function (ls::Vector{Layer{T}})(x::Vector{T}) where T
	for l in ls
		x = l(x)
	end
	return x
end

C(ŷ, y) = 0.5sum(abs2, ŷ .- y)
