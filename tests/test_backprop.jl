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

    @test forward!(DT{Float32,Float32}(), test_layers, x) ≈ test_layers(x)

    ∇′ = _grads(test_layers, x, y)
    d = DT{Float32,Float32}()
    empty!(d)
    ŷ = forward!(d, test_layers, x)
    ((∇3, ∇4), (∇1, ∇2)) = @test_nowarn grads(d, test_layers, ŷ, y)
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