include("types.jl")
include("grads.jl")
using Flux.Optimise: apply!
using Statistics: mean
using LinearAlgebra: norm
using ThreadTools
using LoopVectorization

function forward!(s::DT{T}, l::Layer{T}, ŷ::AbstractVector{T}) where T
    ỹ = muladd(l.W',ŷ,l.b)
    bi = !l.activation .| (ỹ .> 0)
    push!(s, (ξ=ŷ,bi=bi,l=Ref(l)))
    ŷ = ifelse(l.activation, relu.(ỹ), ỹ)
end
function forward!(s::DT{T}, l::ResidualLayer{T}, ŷ::AbstractVector{T}) where T
		ỹ = muladd(l.W',ŷ,l.b)
		bi = !l.activation .| (ỹ .> 0)
		push!(s, (ξ=ŷ,bi=bi,l=Ref(l)))
		ỹ = ifelse(l.activation, relu.(ỹ), ỹ)
        ŷ = l.eta*ŷ + ỹ
end
function forward!(s::DT{T}, ls::Vector{<:AbstractNetworkLayer{T}}, x::AbstractVector{T}) where T
    empty!(s)
	ŷ = x
	for l in ls
        ŷ = forward!(s, l, ŷ)
	end
	ŷ
end

function grads(J::LinearAlgebra.Adjoint{T, Vector{T}}, ∇W, ∇b, ξ::AbstractVector{T}, bi::BitVector, l::Layer{T}, do_jac_computation::Bool) where T
    ∇Wₗ = (J*LazyJac(ξ, bi))'
    push!(∇W,∇Wₗ)
    ∇bₗ = J'.*∂∂b(ξ, bi)
    push!(∇b,∇bₗ)

    if do_jac_computation
        J = J*∂∂ξ(l.W, bi)  # Skip jacobian computation on last layer
    end
    J
end
function grads(J::LinearAlgebra.Adjoint{T, Vector{T}}, ∇W, ∇b, ξ::AbstractVector{T}, bi::BitVector, l::ResidualLayer{T}, do_jac_computation::Bool) where T 
    ∇Wₗ = reshape((J*LazyJac(ξ, bi))', size(l.W))
    push!(∇W,∇Wₗ)
    ∇bₗ = J'.*∂∂b(ξ, bi)
    push!(∇b,∇bₗ)

    if do_jac_computation
        J = J*∂∂ξ(l.W, l.eta, bi)  # Skip jacobian computation on last layer
    end
    J
end
function reverse!(s::DT{T}, layers::Vector{<:AbstractNetworkLayer{T}}, ŷ::AbstractVector{T}, y::AbstractVector{T}) where T
    isempty(s) && error("Stack is empty; did you run the forward pass?")
    @warn "This method uses a lot of memory. Please use reverse_opt! instead."
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
        ŷs::Vector{<:AbstractVector{T}}, 
        ys::Vector{<:AbstractVector{T}}, 
        opts::Vector{Tuple{O,O}},
        normalize::Bool,
    ) where {T,O}
    any(isempty.(stacks)) && error("Stack is empty; did you run the forward pass?")
    Js = [∂C∂(ŷ, y) for (ŷ, y) in zip(ŷs, ys)]
    if normalize
        @debug "Jacobian norm" norm(Js[1])
        Js ./= [√norm(J) for J in Js]  # √x brings x halfway to 1 in log space
        @debug "Jacobian norm after scaling" norm(Js[1])
    end
    for (i, (l, (optW,optb))) in enumerate(zip(reverse(layers), reverse(opts)))
        vals = pop!.(stacks)
        @debug "Jacobian norms layer $(length(layers)-i+1):" J=norm(Js[1]) ∂∂ξ=norm(∂∂ξ(l, vals[1][:bi]))
        # ∇Wₗ = zeros(size(l.W))
        # ∇bₗ = zeros(size(l.b))
        # @inbounds @fastmath for (J, (ξ, bi, _)) in zip(Js, vals)
        #     @show size(∇Wₗ), size(J*LazyJac(ξ, bi))
        #     @show size(∇bₗ), size(J'.*∂∂b(ξ, bi))
        #     @turbo @. ∇Wₗ += J*LazyJac(ξ, bi)
        #     @turbo @. ∇bₗ += J'.*∂∂b(ξ, bi)
        # end
        # @turbo ∇Wₗ ./= length(stacks)
        # @turbo ∇bₗ ./= length(stacks)
        ∇Wₗ = mean(tmap(zip(Js, vals)) do (J, (ξ, bi, _))
            @fastmath J*LazyJac(ξ, bi)
        end)  # TODO: Mean accum takes a lot of time
        ΔW = apply!(optW, l.W, ∇Wₗ)
        @inbounds @fastmath l.W .= l.W - ΔW
        ∇bₗ = mean(tmap(zip(Js, vals)) do (J, (ξ, bi, _))
            # TODO: Combine loops
            @fastmath J'.*∂∂b(ξ, bi)
        end)
        @inbounds @fastmath l.b .= l.b - apply!(optb, l.b, ∇bₗ)

		if !isempty(first(stacks))
			Js = [  # Skip jacobian computation on last layer
                J*∂∂ξ(refl[], bi)  # TODO: LazyJac?
                for (J, (_, bi, refl)) in zip(Js, vals)
            ]
		end
    end
end

# opts = [(ADAM(), ADAM()) for l in layers]
function train_batch!(ds::Vector{DT{T}}, layers::Vector{<:AbstractNetworkLayer{T}}, xs::Vector{<:AbstractVector{T}}, ys::Vector{<:AbstractVector{T}}, opts, normalize::Bool = false) where T
    ŷs = similar(ys)
    n = length(xs)
    @threads for i in 1:n  # TODO: Parallelize
        ŷs[i] = forward!(ds[i], layers, xs[i])
    end
    reverse_opt!(ds, layers, ŷs, ys, opts, normalize)
    mean(C(layers(x), y) for (x,y) in zip(xs,ys))
end

function train_batch!(layers::Vector{<:AbstractNetworkLayer{T}}, xs::Vector{<:AbstractVector{T}}, ys::Vector{<:AbstractVector{T}}, opts, normalize::Bool = false) where T
    ds = [DT{T}() for _ in 1:length(xs)]
    train_batch!(ds, layers, xs, ys, opts, normalize)
end
