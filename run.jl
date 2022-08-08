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

network = [
    Layer(in_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>hidden_dim)
    Layer(hidden_dim=>out_dim, false)
]
N = 4
xs = [randn(Float32, in_dim) for _ in 1:N]
ys = [randn(Float32, out_dim) for _ in 1:N]

opts = [(Descent(1f-3), Descent(1f-3)) for l in network]
@profview @btime (
    @show train_batch!($network, $xs, $ys, $opts, false)
) seconds = 60
