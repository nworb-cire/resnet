include("../layers.jl")
include("../grads.jl")
include("../lazyjacobian.jl")
include("../training.jl")
using Flux: Descent, update!
in_dim = 3
hidden_dim = 4
out_dim = 2

import Random: seed!
seed!(123)

L1 = Layer(in_dim=>hidden_dim)
L2 = Layer(hidden_dim=>out_dim, false)
test_layers = (L1, L2)
test_network = Network(test_layers, in_dim, out_dim, hidden_dim)

x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

using Test
using Zygote

function _grads(ls::NTuple{2,Layer{T}}, x::Vector{T}, y::Vector{T}) where T
    L1, L2 = ls

    y1 = L1(x)
    y2 = L2(y1)
    
    J1 = ∂C∂(y2, y)
    
    ∇2 = Array(J1*LazyJac(y1, _∂∂b(L2, y1)))
    ∇2 = reshape(∇2, size(L2.W))
    ∇b2 = J1'.*∂∂b(L2, y1)
    
    J2 = ∂∂ξ(L2, y1)
    J1J2 = J1*J2
    
    ∇1 = Array(J1J2*LazyJac(x, _∂∂b(L1, x)))
    ∇1 = reshape(∇1, size(L1.W))
    ∇b1 = J1J2'.*∂∂b(L1, x)

    (∇1, ∇b1, ∇2, ∇b2)
end

@testset "feedforward network gradients" begin
    @test forward!(DT{Float32}(), test_network, x) ≈ test_network(x)

    ∇1, ∇b1, ∇2, ∇b2 = _grads(test_layers, x, y)
    @test ∇1 ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(ŷ), y)
    end[1]
    @test ∇b1 ≈ gradient(L1.b) do b
        ŷ = muladd(L1.W',x,b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(ŷ), y)
    end[1]
    @test ∇2 ≈ gradient(L2.W) do W
        ŷ = muladd(W',L1(x),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
    @test ∇b2 ≈ gradient(L2.b) do b
        ŷ = muladd(L2.W',L1(x),b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end;

@testset "feedforward reverse opt" begin
    N = 100
    xs = [randn(Float32, in_dim) for _ in 1:100]
    ys = [randn(Float32, out_dim) for _ in 1:100]
    opts = [(Descent(), Descent()) for l in test_layers]
    baseline_loss = mean(C(test_network(x), y) for (x,y) in zip(xs,ys))
    for _ in 1:100 train_batch!(test_network, xs, ys, opts) end
    final_loss = mean(C(test_network(x), y) for (x,y) in zip(xs,ys))
    @test final_loss < baseline_loss
end;

