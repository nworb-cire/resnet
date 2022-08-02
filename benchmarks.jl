using BenchmarkTools

include("layers.jl")
include("grads.jl")

in_dim = 30
hidden_dim = 120
out_dim = 6

import Random: seed!
seed!(123)

network = [
    Layer(in_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>out_dim)
]
x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

d = DT{Float32,Float32}()
@benchmark begin
    forward!($d, $network, $x, $y)
end
@benchmark begin
    forward!($d, $network, $x, $y)  # Need to populate stack
    reverse!($d, $network, $x, $y)
end

# function train!(d::DT{T,S}, layers::Vector{Layer{T}}, x::Vector{T}, y::Vector{T}, η::T) where {S,T}
# 	empty!(d)
# 	# overhead = Base.summarysize(d)
# 	ŷ = forward!(d, layers, x, y)
# 	# memusage = Base.summarysize(d) - overhead
# 	# @info "Memory usage: $memusage"
# 	∇ = grads(d, layers, x, y)
# 	for ((∇W, ∇b), l) in zip(∇, reverse(layers))
# 		l.W .-= η*∇W
# 		l.b .-= η*∇b
# 	end
# 	C(layers(x), y)
# end

# η = 1f-2
