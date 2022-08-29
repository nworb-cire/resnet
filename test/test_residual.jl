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
R1 = ResidualLayer(hidden_dim=>hidden_dim)
L2 = Layer(hidden_dim=>out_dim, false)
test_residual_layers = (L1, R1, L2)
test_residual_network = Network(test_residual_layers, in_dim, out_dim, hidden_dim)

x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

using Test
using Zygote


function _grads(ls::Tuple{Vararg{<:AbstractNetworkLayer{T},3}}, x::Vector{T}, y::Vector{T}) where T
    L1, R1, L2 = ls

    y1 = L1(x)
    y2 = R1(y1)
    y3 = L2(y2)
    
    J1 = ∂C∂(y3, y)
    
    ∇3 = Array(J1*LazyJac(y2, _∂∂b(L2, y2)))
    ∇b3 = J1'.*∂∂b(L2, y2)
    
    J2 = ∂∂ξ(L2, y2)
    J1J2 = J1*J2

    ∇2 = Array(J1J2*LazyJac(y1, _∂∂b(R1, y1)))
    ∇b2 = J1J2'.*∂∂b(R1, y2)

    J3 = ∂∂ξ(R1, y1)
    J1J2J3 = J1J2*J3
    
    ∇1 = Array(J1J2J3*LazyJac(x, _∂∂b(L1, x)))
    ∇b1 = J1J2J3'.*∂∂b(L1, x)

    (∇1, ∇b1, ∇2, ∇b2, ∇3, ∇b3)
end

@testset "residual network gradients" begin
    @test forward!(DT{Float32}(), test_residual_network, x) ≈ test_residual_network(x)

    ∇1, ∇b1, ∇2, ∇b2, ∇3, ∇b3 = _grads(test_residual_layers, x, y)
    @test ∇1 ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(R1(ŷ)), y)
    end[1]
    @test ∇b1 ≈ gradient(L1.b) do b
        ŷ = muladd(L1.W',x,b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(R1(ŷ)), y)
    end[1]
    @test ∇2 ≈ gradient(R1.W) do W
        ŷ = muladd(W',L1(x),R1.b)
        ŷ = ifelse(R1.activation, relu.(ŷ), ŷ)
        ŷ = R1.eta*L1(x) + ŷ
        C(L2(ŷ), y)
    end[1]
    @test ∇b2 ≈ gradient(R1.b) do b
        ŷ = muladd(R1.W',L1(x),b)
        ŷ = ifelse(R1.activation, relu.(ŷ), ŷ)
        ŷ = R1.eta*L1(x) + ŷ
        C(L2(ŷ), y)
    end[1]
    @test ∇3 ≈ gradient(L2.W) do W
        ŷ = muladd(W',R1(L1(x)),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
    @test ∇b3 ≈ gradient(L2.b) do b
        ŷ = muladd(L2.W',R1(L1(x)),b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end;

@testset "residual reverse opt" begin
    N = 100
    xs = [randn(Float32, in_dim) for _ in 1:100]
    ys = [randn(Float32, out_dim) for _ in 1:100]
    opts = [(Descent(), Descent()) for l in test_residual_layers]
    baseline_loss = mean(C(test_residual_network(x), y) for (x,y) in zip(xs,ys))
    for _ in 1:100 train_batch!(test_residual_network, xs, ys, opts) end
    final_loss = mean(C(test_residual_network(x), y) for (x,y) in zip(xs,ys))
    @test final_loss < baseline_loss
end;