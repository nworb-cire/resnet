include("../layers.jl")
include("../grads.jl")
include("../lazyjacobian.jl")
include("../training.jl")
using Flux: ADAM
in_dim = 3
hidden_dim = 4
out_dim = 2

import Random: seed!
seed!(123)

L1 = Layer(in_dim=>hidden_dim)
L2 = Layer(hidden_dim=>out_dim, false)
test_layers = [L1, L2]

R1 = ResidualLayer(hidden_dim=>hidden_dim)
test_residual_network = [L1, R1, L2]

x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

using Test
using Zygote

@testset "feedforward network gradients" begin
    function _grads(ls::Vector{Layer{T}}, x::Vector{T}, y::Vector{T}) where T
        L1, L2 = ls
    
        y1 = L1(x)
        y2 = L2(y1)
        
        J1 = ∂C∂(y2, y)
        
        ∇2 = J1*∂∂W(L2, y1)
        ∇2 = reshape(∇2, size(L2.W))
        ∇b2 = J1'.*∂∂b(L2, y1)
        
        J2 = ∂∂ξ(L2, y1)
        J1J2 = J1*J2
        
        ∇1 = J1J2*∂∂W(L1, x)
        ∇1 = reshape(∇1, size(L1.W))
        ∇b1 = J1J2'.*∂∂b(L1, x)
    
        (∇1, ∇b1, ∇2, ∇b2)
    end

    @test forward!(DT{Float32}(), test_layers, x) ≈ test_layers(x)

    ∇′ = _grads(test_layers, x, y)
    d = DT{Float32}()
    empty!(d)
    ŷ = forward!(d, test_layers, x)
    ((∇3, ∇4), (∇1, ∇2)) = @test_nowarn reverse!(d, test_layers, ŷ, y)
    empty!(d)

    @test ∇1' ≈ ∇′[1]
    @test ∇1' ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        if L1.activation
            C(L2(relu.(ŷ)), y)
        else
            C(L2(ŷ), y)
        end
    end[1]
    @test ∇2 ≈ ∇′[2]
    @test ∇3' ≈ ∇′[3]
    @test ∇3' ≈ gradient(L2.W) do W
        ŷ = muladd(W',L1(x),L2.b)
        if L2.activation
            C(relu.(ŷ), y)
        else
            C(ŷ, y)
        end
    end[1]
    @test ∇4 ≈ ∇′[4]
end;

@testset "feedforward reverse opt" begin
    N = 100
    xs = [randn(Float32, in_dim) for _ in 1:100]
    ys = [randn(Float32, out_dim) for _ in 1:100]
    opts = [(ADAM(), ADAM()) for l in test_layers]
    baseline_loss = mean(C(test_layers(x), y) for (x,y) in zip(xs,ys))
    for _ in 1:100 train_batch!(test_layers, xs, ys, opts) end
    final_loss = mean(C(test_layers(x), y) for (x,y) in zip(xs,ys))
    @test final_loss < baseline_loss
end;

@testset "residual network gradients" begin
    function _grads(ls::Vector{<:AbstractNetworkLayer{T}}, x::Vector{T}, y::Vector{T}) where T
        L1, R1, L2 = ls
    
        y1 = L1(x)
        y2 = R1(y1)
        y3 = L2(y2)
        
        J1 = ∂C∂(y3, y)
        
        ∇3 = J1*∂∂W(L2, y2)
        ∇3 = reshape(∇3, size(L2.W))
        ∇b3 = J1'.*∂∂b(L2, y2)
        
        J2 = ∂∂ξ(L2, y2)
        J1J2 = J1*J2

        ∇2 = J1J2*∂∂W(R1, y1)
        ∇2 = reshape(∇2, size(R1.W))
        ∇b2 = J1J2'.*∂∂b(R1, y1)

        J3 = ∂∂ξ(R1, y1)
        J1J2J3 = J1J2*J3
        
        ∇1 = J1J2J3*∂∂W(L1, x)
        ∇1 = reshape(∇1, size(L1.W))
        ∇b1 = J1J2J3'.*∂∂b(L1, x)
    
        (∇1, ∇b1, ∇2, ∇b2, ∇3, ∇b3)
    end

    @test forward!(DT{Float32}(), test_residual_network, x) ≈ test_residual_network(x)

    ∇′ = _grads(test_residual_network, x, y)
    d = DT{Float32}()
    empty!(d)
    ŷ = forward!(d, test_residual_network, x)
    ((∇5, ∇6), (∇3, ∇4), (∇1, ∇2)) = @test_nowarn reverse!(d, test_residual_network, ŷ, y)
    empty!(d)

    @test ∇1' ≈ ∇′[1]
    @test ∇1' ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        if L1.activation
            C(L2(R1(relu.(ŷ))), y)
        else
            C(L2(R1(ŷ)), y)
        end
    end[1]
    @test ∇2 ≈ ∇′[2]
    @test ∇3' ≈ ∇′[3]
    @test ∇3' ≈ gradient(R1.W) do W
        ŷ = L1(x)
        C(L2(R1.eta*ŷ + relu.(muladd(W',ŷ,R1.b))), y)
    end[1]
    @test ∇4 ≈ ∇′[4]
    @test ∇5' ≈ ∇′[5]
    @test ∇5' ≈ gradient(L2.W) do W
        ŷ = muladd(W',R1(L1(x)),L2.b)
        if L2.activation
            C(relu.(ŷ), y)
        else
            C(ŷ, y)
        end
    end[1]
    @test ∇6 ≈ ∇′[6]
end;

@testset "residual reverse opt" begin
    N = 100
    xs = [randn(Float32, in_dim) for _ in 1:100]
    ys = [randn(Float32, out_dim) for _ in 1:100]
    opts = [(ADAM(), ADAM()) for l in test_residual_network]
    baseline_loss = mean(C(test_residual_network(x), y) for (x,y) in zip(xs,ys))
    for _ in 1:100 train_batch!(test_residual_network, xs, ys, opts) end
    final_loss = mean(C(test_residual_network(x), y) for (x,y) in zip(xs,ys))
    @test final_loss < baseline_loss
end;
