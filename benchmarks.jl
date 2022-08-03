using BenchmarkTools
using Flux: ADAM

include("layers.jl")
include("grads.jl")
include("training.jl")

in_dim = 4000  # 4_000
hidden_dim = 1200  # 12_000
out_dim = 221  # 221

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
N = 1
xs = [randn(Float32, in_dim) for _ in 1:100]
ys = [randn(Float32, out_dim) for _ in 1:100]
opts = [(ADAM(), ADAM()) for l in network]

# Compile
begin
    d = DT{Float32,Float32}()
    ŷ = forward!(d, network, xs[1])
    reverse_opt!([d], network, [ŷ], [ys[1]], opts)
end

# @profview forward!(DT{Float32,Float32}(), network, xs[1])
# @benchmark forward!($DT{Float32,Float32}(), $network, $(xs[1]))
# @profview begin
#     d = DT{Float32,Float32}()
#     ŷ = forward!(d, network, xs[1])
#     reverse_opt!([d], network, [ŷ], [ys[1]], opts)
# end
# @benchmark begin
#     d = DT{Float32,Float32}()
#     ŷ = forward!(d, $network, $xs[1])
#     reverse_opt!([d], $network, [ŷ], [$ys[1]], $opts)
# end
