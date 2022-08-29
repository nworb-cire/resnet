include("../layers.jl")
include("../grads.jl")
include("../lazyjacobian.jl")
in_dim = 3
hidden_dim = 4
out_dim = 2

import Random: seed!
seed!(123)

L1 = Layer(in_dim=>hidden_dim)
L2 = Layer(hidden_dim=>out_dim, false)
test_layers = (L1, L2)
test_network = Network(test_layers, in_dim, out_dim, hidden_dim)

R1 = ResidualLayer(hidden_dim=>hidden_dim)
test_residual_layers = (L1, R1, L2)
test_residual_network = Network(test_residual_layers, in_dim, out_dim, hidden_dim)

x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

using Test
using Zygote

@testset "weight jacobians" begin
    @test ∂∂W(L1, x) ≈ jacobian(L1.W) do W 
        relu.(muladd(W',x,L1.b))
    end[1]
    @test ∂∂W(R1, L1(x)) ≈ jacobian(R1.W) do W 
        ŷ = muladd(W',L1(x),R1.b)
        R1.eta*L1(x) + relu.(ŷ)
    end[1]
end

@testset "bias jacobians" begin
    # Note that Zygote returns a matrix but we multiply entrywise.
    @test Diagonal(∂∂b(L1, x)) ≈ jacobian(L1.b) do b
        relu.(muladd(L1.W',x,b))
    end[1]
    @test Diagonal(∂∂b(R1, L1(x))) ≈ jacobian(R1.b) do b
        ŷ = muladd(R1.W',L1(x),b)
        R1.eta*L1(x) + relu.(ŷ)
    end[1]
end

@testset "passthrough jacobians" begin
    @test ∂∂ξ(L1, x) ≈ jacobian(L1, x)[1]
    @test ∂∂ξ(L1, x) ≈ ∂∂ξ(L1.W, _∂∂b(L1, x))
    @test ∂∂ξ(L1, x) ≈ ∂∂ξ(L1, _∂∂b(L1, x))

    ŷ = L1(x)
    @test ∂∂ξ(R1, ŷ) ≈ jacobian(R1, ŷ)[1]
    @test ∂∂ξ(R1, ŷ) ≈ ∂∂ξ(R1.W, R1.eta, _∂∂b(R1, ŷ))
    @test ∂∂ξ(R1, ŷ) ≈ ∂∂ξ(R1, _∂∂b(R1, ŷ))
end

@testset "cost jacobians" begin
    @test ∂C∂(test_network(x), y) ≈ jacobian(x->C(x,y), test_network(x))[1]
    @test ∂C∂(test_residual_network(x), y) ≈ jacobian(x->C(x,y), test_residual_network(x))[1]
end

@testset "feedforward weight gradients" begin
    @test reshape(
        ∂C∂(test_network(x), y)*∂∂ξ(L2, L1(x))*∂∂W(L1, x),
        size(L1.W)
     ) ≈ gradient(L1.W) do W
        C(L2(relu.(muladd(W',x,L1.b))), y)
    end[1]
    @test reshape(
        ∂C∂(test_network(x), y)*∂∂W(L2, L1(x)),
        size(L2.W)
     ) ≈ gradient(L2.W) do W
        ŷ = muladd(W',L1(x),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end

@testset "lazy feedforward weight gradients" begin
    @test Array(∂C∂(test_network(x), y)*∂∂ξ(L2, L1(x))*LazyJac(x, _∂∂b(L1, x))) ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(ŷ), y)
    end[1]
    @test Array(∂C∂(test_network(x), y)*LazyJac(L1(x), _∂∂b(L2, L1(x)))) ≈ gradient(L2.W) do W
        ŷ = muladd(W',L1(x),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end

@testset "residual weight gradients" begin
    @test reshape(
        ∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x)))*∂∂ξ(R1, L1(x))*∂∂W(L1, x),
        size(L1.W)
     ) ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(R1(ŷ)), y)
    end[1]
    @test reshape(
        ∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x)))*∂∂W(R1, L1(x)),
        size(R1.W)
     ) ≈ gradient(R1.W) do W
        ŷ = muladd(W',L1(x),R1.b)
        ŷ = ifelse(R1.activation, relu.(ŷ), ŷ)
        ŷ = R1.eta*L1(x) + ŷ
        C(L2(ŷ), y)
    end[1]
    @test reshape(
        ∂C∂(test_residual_network(x), y)*∂∂W(L2, R1(L1(x))),
        size(L2.W)
     ) ≈ gradient(L2.W) do W
        ŷ = muladd(W',R1(L1(x)),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end

@testset "lazy residual weight gradients" begin
    @test Array(∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x)))*∂∂ξ(R1, L1(x))*LazyJac(x, _∂∂b(L1, x))) ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(R1(ŷ)), y)
    end[1]
    @test Array(∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x)))*LazyJac(L1(x), _∂∂b(R1, L1(x)))) ≈ gradient(R1.W) do W
        ŷ = muladd(W',L1(x),R1.b)
        ŷ = ifelse(R1.activation, relu.(ŷ), ŷ)
        ŷ = R1.eta*L1(x) + ŷ
        C(L2(ŷ), y)
    end[1]
    @test Array(∂C∂(test_residual_network(x), y)*LazyJac(R1(L1(x)), _∂∂b(L2, R1(L1(x))))) ≈ gradient(L2.W) do W
        ŷ = muladd(W',R1(L1(x)),L2.b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end

@testset "feedforward bias gradients" begin
    @test (∂C∂(test_network(x), y)*∂∂ξ(L2, L1(x)))'.*∂∂b(L1, x) ≈ gradient(L1.b) do b
        C(L2(relu.(muladd(L1.W',x,b))), y)
    end[1]
    @test ∂C∂(test_network(x), y)'.*∂∂b(L2, L1(x)) ≈ gradient(L2.b) do b
        ŷ = muladd(L2.W',L1(x),b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end

@testset "residual bias gradients" begin
    @test (∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x)))*∂∂ξ(R1, L1(x)))'.*∂∂b(L1, x) ≈ gradient(L1.b) do b
        ŷ = muladd(L1.W',x,b)
        ŷ = ifelse(L1.activation, relu.(ŷ), ŷ)
        C(L2(R1(ŷ)), y)
    end[1]
    @test (∂C∂(test_residual_network(x), y)*∂∂ξ(L2, R1(L1(x))))'.*∂∂b(R1, L1(x)) ≈ gradient(R1.b) do b
        ŷ = muladd(R1.W',L1(x),b)
        ŷ = ifelse(R1.activation, relu.(ŷ), ŷ)
        ŷ = R1.eta*L1(x) + ŷ
        C(L2(ŷ), y)
    end[1]
    @test ∂C∂(test_residual_network(x), y)'.*∂∂b(L2, R1(L1(x))) ≈ gradient(L2.b) do b
        ŷ = muladd(L2.W',R1(L1(x)),b)
        ŷ = ifelse(L2.activation, relu.(ŷ), ŷ)
        C(ŷ, y)
    end[1]
end
