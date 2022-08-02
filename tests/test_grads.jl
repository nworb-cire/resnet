include("../layers.jl")
include("../grads.jl")
in_dim = 3
hidden_dim = 4
out_dim = 2

import Random: seed!
seed!(123)

L1 = Layer(in_dim=>hidden_dim)
L2 = Layer(hidden_dim=>out_dim, false)
test_layers = [L1, L2]

x = randn(Float32, in_dim)
y = randn(Float32, out_dim)

using Test
using Zygote

@testset "feedforward layer jacobians and gradients" begin
    @testset "passthrough jacobians" begin
        @test ∂∂ξ(L2, L1(x)) * ∂∂ξ(L1, x) ≈ jacobian(L2∘L1, x)[1]
        @test ∂∂ξ(L1, x) ≈ jacobian(L1, x)[1]
        @test ∂∂ξ(L2, L1(x)) * ∂∂W(L1, x) ≈ jacobian(L1.W) do W
            L2(relu.(muladd(W',x,L1.b)))
        end[1]
        @test ∂C∂(L2(L1(x)), y)*∂∂ξ(L2, L1(x)) ≈ gradient(L1(x)) do ξ
            C(L2(ξ), y)
        end[1]'
    end

    @testset "weight jacobians" begin
        @test ∂∂W(L1, x) ≈ jacobian(L1.W) do W
            relu.(W'x.+L1.b)
        end[1]
        @test ∂∂W(L2, L1(x)) ≈ jacobian(L2.W) do W
            ŷ = muladd(W',L1(x),L2.b)
            if L2.activation
                relu.(ŷ)
            else
                ŷ
            end
        end[1]
    end

    @testset "bias jacobians" begin
    # Note that Zygote returns a matrix but we multiply entrywise.
        @test Diagonal(∂∂b(L1, x)) ≈ jacobian(L1.b) do b
            relu.(muladd(L1.W',x,b))
        end[1]
        @test Diagonal(∂∂b(L2, L1(x))) ≈ jacobian(L2.b) do b
            y = muladd(L2.W',L1(x),b)
            if L2.activation
                relu.(y)
            else
                y
            end
        end[1]
    end

    @testset "passthrough weight" begin
        @test ∂∂ξ(L2, L1(x))*∂∂W(L1, x) ≈ jacobian(L1.W) do W
            L2(relu.(muladd(W',x,L1.b)))
        end[1]
        @test ∂∂W(L2, L1(x)) ≈ jacobian(L2.W) do W
            ŷ = muladd(W',L1(x),L2.b)
            if L2.activation
                relu.(ŷ)
            else
                ŷ
            end
        end[1]
    end

    @testset "passthrough bias" begin
        @test ∂∂ξ(L2, L1(x))*Diagonal(∂∂b(L1, x)) ≈ jacobian(L1.b) do b
            L2(relu.(muladd(L1.W',x,b)))
        end[1]
    end

    @testset "weight gradients" begin
        @test reshape(
            ∂C∂(test_layers(x), y)*∂∂ξ(L2, L1(x))*∂∂W(L1, x), size(L1.W)
        ) ≈ gradient(L1.W) do W
            C(L2(relu.(muladd(W',x,L1.b))), y)
        end[1]
        @test reshape(
            ∂C∂(test_layers(x), y)*∂∂W(L2, L1(x)), size(L2.W)
        ) ≈ gradient(L2.W) do W
            ŷ = muladd(W',L1(x),L2.b)
            if L2.activation
                C(relu.(ŷ), y)
            else
                C(ŷ, y)
            end
        end[1]
    end

    @testset "bias gradients" begin
        @test (∂C∂(test_layers(x), y)*∂∂ξ(L2, L1(x)))'.*∂∂b(L1, x) ≈ gradient(L1.b) do b
            C(L2(relu.(muladd(L1.W',x,b))), y)
        end[1]
        @test ∂C∂(test_layers(x), y)'.*∂∂b(L2, L1(x)) ≈ gradient(L2.b) do b
            ŷ = muladd(L2.W',L1(x),b)
            if L2.activation
                C(relu.(ŷ), y)
            else
                C(ŷ, y)
            end
        end[1]
    end
end;

@testset "feedforward compressed jacobians and gradients" begin
    @testset "passthrough jacobians" begin
        @test ∂∂ξ(L2, _∂∂b(L2,L1(x))) ≈ ∂∂ξ(L2, L1(x))
        @test ∂∂ξ(L2.W, _∂∂b(L2,L1(x))) ≈ ∂∂ξ(L2, L1(x))
    end
    
    @testset "weight jacobians" begin
        @test ∂∂W(L1(x), _∂∂b(L2,L1(x))) ≈ ∂∂W(L2, L1(x))
        @test ∂∂W(x, _∂∂b(L1,x)) ≈ ∂∂W(L1, x)
    end
    
    @testset "bias jacobians" begin
        @test ∂∂b(L1(x), _∂∂b(L2,L1(x))) ≈ ∂∂b(L2, L1(x))
        @test ∂∂b(x, _∂∂b(L1,x)) ≈ ∂∂b(L1, x)
    end
    
    @testset "passthrough weight" begin
        @test ∂∂ξ(L2.W, _∂∂b(L2,L1(x)))*∂∂W(x, _∂∂b(L1,x)) ≈ jacobian(L1.W) do W
            L2(relu.(muladd(W',x,L1.b)))
        end[1]
        @test ∂∂W(L1(x), _∂∂b(L2,L1(x))) ≈ jacobian(L2.W) do W
            ŷ = muladd(W',L1(x),L2.b)
            if L2.activation
                relu.(ŷ)
            else
                ŷ
            end
        end[1]
    end
    
    @testset "passthrough bias" begin
        @test ∂∂ξ(L2.W, _∂∂b(L2,L1(x)))*Diagonal(∂∂b(x, _∂∂b(L1,x))) ≈ jacobian(L1.b) do b
            L2(relu.(muladd(L1.W',x,b)))
        end[1]
    end
    
    @testset "weight gradients" begin
        @test ∂C∂(test_layers(x), y)*∂∂ξ(L2.W, _∂∂b(L2,L1(x)))*LazyJac(x, _∂∂b(L1,x)) ≈ gradient(L1.W) do W
            C(L2(relu.(muladd(W',x,L1.b))), y)
        end[1]'
        @test ∂C∂(test_layers(x), y)*LazyJac(L1(x), _∂∂b(L2,L1(x))) ≈ gradient(L2.W) do W
            ŷ = muladd(W',L1(x),L2.b)
            if L2.activation
                C(relu.(ŷ), y)
            else
                C(ŷ, y)
            end
        end[1]'
    end
    
    @testset "bias gradients" begin
        @test (∂C∂(test_layers(x), y)*∂∂ξ(L2.W, _∂∂b(L2,L1(x))))'.*∂∂b(x, _∂∂b(L1,x)) ≈ gradient(L1.b) do b
            C(L2(relu.(muladd(L1.W',x,b))), y)
        end[1]
        @test ∂C∂(test_layers(x), y)'.*∂∂b(L1(x), _∂∂b(L2,L1(x))) ≈ gradient(L2.b) do b
            ŷ = muladd(L2.W',L1(x),b)
            if L2.activation
                C(relu.(ŷ), y)
            else
                C(ŷ, y)
            end
        end[1]
    end
end;

@testset "network gradients" begin
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

    @test forward!(DT{Float32,Float32}(), test_layers, x, y) ≈ C(test_layers(x), y)

    ∇′ = _grads(test_layers, x, y)
    d = DT{Float32,Float32}()
    empty!(d)
    ŷ = forward!(d, test_layers, x, y)
    ((∇3, ∇4), (∇1, ∇2)) = grads(d, test_layers, x, y)
    empty!(d)

    @test ∇1 ≈ ∇′[1]
    @test ∇1 ≈ gradient(L1.W) do W
        ŷ = muladd(W',x,L1.b)
        if L1.activation
            C(L2(relu.(ŷ)), y)
        else
            C(L2(ŷ), y)
        end
    end[1]
    @test ∇2 ≈ ∇′[2]
    @test ∇3 ≈ ∇′[3]
    @test ∇3 ≈ gradient(L2.W) do W
        ŷ = muladd(W',L1(x),L2.b)
        if L2.activation
            C(relu.(ŷ), y)
        else
            C(ŷ, y)
        end
    end[1]
    @test ∇4 ≈ ∇′[4]
end;
