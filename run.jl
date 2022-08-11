import Random: seed!
seed!(123)

using BenchmarkTools
using Flux: ADAM, Descent, Momentum

include("layers.jl")
include("grads.jl")
include("training.jl")

in_dim = 400
hidden_dim = 1200
out_dim = 22

network::Vector{<:AbstractNetworkLayer{Float32}} = [
    Layer(in_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    ResidualLayer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>out_dim, false)
]
N = 64
xs::Vector{Vector{Float32}} = [randn(Float32, in_dim) for _ in 1:N]
ys::Vector{Vector{Float32}} = [randn(Float32, out_dim) for _ in 1:N]

opts = [(Descent(1f-4), Descent(1f-4)) for l in network]
# @profview @btime (
#     @show train_batch!($network, $xs, $ys, $opts, true)
# ) seconds = 10
# @profview @time for _ in 1:10
#     train_batch!(network, xs, ys, opts, true)
# end
# @profview @time for _ in 1:1000
#     forward!(
#         DT{Float32}(), 
#         network::Vector{<:AbstractNetworkLayer{Float32}}, 
#         xs[1]::Vector{Float32}
#     )
# end
