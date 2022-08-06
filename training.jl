include("grads.jl")
using Flux.Optimise: apply!
using Statistics: mean
using LinearAlgebra: norm

if !@isdefined DT
    DT{T} = Deque{
        NamedTuple{(:ξ, :bi, :l), Tuple{
            Vector{T}, 
        Vector{T}, 
            Vector{T}, 
            BitVector, 
        BitVector, 
            BitVector, 
            Base.RefValue{<:AbstractNetworkLayer{T}}
        }}
    } where T
end

function forward!(s::DT{T}, l::Layer{T}, ŷ::Vector{T}) where T
    ỹ = muladd(l.W',ŷ,l.b)
    bi = !l.activation .| (ỹ .> 0)
    push!(s, (ξ=ŷ,bi=bi,l=Ref(l)))
    ŷ = ifelse(l.activation, relu.(ỹ), ỹ)
end
function forward!(s::DT{T}, l::ResidualLayer{T}, ŷ) where T
		ỹ = muladd(l.W',ŷ,l.b)
		bi = !l.activation .| (ỹ .> 0)
		push!(s, (ξ=ŷ,bi=bi,l=Ref(l)))
		ỹ = ifelse(l.activation, relu.(ỹ), ỹ)
        ŷ = l.eta*ŷ + ỹ
end
function forward!(s::DT{T}, ls::Vector{<:AbstractNetworkLayer{T}}, x::Vector{T}) where T
    empty!(s)
	ŷ = x
	for l in ls
        ŷ = forward!(s, l, ŷ)
	end
	ŷ
end

function grads(J::LinearAlgebra.Adjoint{T, Vector{T}}, ∇W, ∇b, ξ::Vector{T}, bi::BitVector, l::Layer{T}, do_jac_computation::Bool) where T
    ∇Wₗ = (J*LazyJac(ξ, bi))'
    push!(∇W,∇Wₗ)
    ∇bₗ = J'.*∂∂b(ξ, bi)
    push!(∇b,∇bₗ)

    if do_jac_computation
        J = J*∂∂ξ(l.W, bi)  # Skip jacobian computation on last layer
    end
    J
end
function grads(J::LinearAlgebra.Adjoint{T, Vector{T}}, ∇W, ∇b, ξ::Vector{T}, bi::BitVector, l::ResidualLayer{T}, do_jac_computation::Bool) where T 
    ∇Wₗ = reshape((J*LazyJac(ξ, bi))', size(l.W))
    push!(∇W,∇Wₗ)
    ∇bₗ = J'.*∂∂b(ξ, bi)
    push!(∇b,∇bₗ)

    if do_jac_computation
        J = J*∂∂ξ(l.W, l.eta, bi)  # Skip jacobian computation on last layer
    end
    J
end
function reverse!(s::DT{T}, layers::Vector{<:AbstractNetworkLayer{T}}, ŷ::Vector{T}, y::Vector{T}) where T
    isempty(s) && error("Stack is empty; did you run the forward pass?")
	J = ∂C∂(ŷ, y)
	∇W = Matrix{T}[]
	∇b = Vector{T}[]
	while !isempty(s)
        ξ, bi, refl = pop!(s)
        l = refl[]
		J = grads(J, ∇W, ∇b, ξ, bi, l, !isempty(s))
	end
	return zip(∇W, ∇b)
end

function reverse_opt!(
        stacks::Vector{DT{T}}, 
        layers::Vector{<:AbstractNetworkLayer{T}}, 
        ŷs::Vector{Vector{T}}, 
        ys::Vector{Vector{T}}, 
        opts::Vector{Tuple{O,O}},
        normalize::Bool,
    ) where {T,O}
    any(isempty.(stacks)) && error("Stack is empty; did you run the forward pass?")
    Js = [∂C∂(ŷ, y) for (ŷ, y) in zip(ŷs, ys)]
    @debug "Jacobian norm" norm(Js[1])
    if normalize
        Js ./= [norm(J) for J in Js]
    end
    for (i, (l, (optW,optb))) in enumerate(zip(reverse(layers), reverse(opts)))
        vals = pop!.(stacks)
        @debug "Jacobian norms layer $(length(layers)-i+1):" J=norm(Js[1]) ∂∂ξ=norm(∂∂ξ(l, vals[1][:bi]))
        ∇Wₗ = mean([
            J*LazyJac(ξ, bi)
            for (J, (ξ, bi, _)) in zip(Js, vals)
        ])
        l.W .-= apply!(optW, l.W, ∇Wₗ)
        ∇bₗ = mean([  # TODO: Combine for loops
            J'.*∂∂b(ξ, bi)
            for (J, (ξ, bi, _)) in zip(Js, vals)
        ])
        l.b .-= apply!(optb, l.b, ∇bₗ)

        # ∇Wₗ = zero(l.W')
        # ∇bₗ = zero(l.b)
        # @inbounds @fastmath for (J, (ξ, bi, _)) in zip(Js, vals)
        #     ∇Wₗ .+= J*LazyJac(ξ, bi)
        #     ∇bₗ .+= J'.*∂∂b(ξ, bi)
        # end

		if !isempty(first(stacks))
			Js = [  # Skip jacobian computation on last layer
                J*∂∂ξ(refl[], bi)
                for (J, (_, bi, refl)) in zip(Js, vals)
            ]
		end
    end
end

# opts = [(ADAM(), ADAM()) for l in layers]
function train_batch!(ds::Vector{DT{T}}, layers::Vector{<:AbstractNetworkLayer{T}}, xs::Vector{Vector{T}}, ys::Vector{Vector{T}}, opts, normalize::Bool = false) where T
    ŷs = similar(ys)
    n = length(xs)
    for i in 1:n  # TODO: Parallelize
        ŷs[i] = forward!(ds[i], layers, xs[i])
    end
    reverse_opt!(ds, layers, ŷs, ys, opts, normalize)
    mean(C(layers(x), y) for (x,y) in zip(xs,ys))
end

function train_batch!(layers::Vector{<:AbstractNetworkLayer{T}}, xs::Vector{Vector{T}}, ys::Vector{Vector{T}}, opts, normalize::Bool = false) where T
    ds = [DT{T}() for _ in 1:length(xs)]
    train_batch!(ds, layers, xs, ys, opts, normalize)
end
