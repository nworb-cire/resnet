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
function forward!(s::DT{T}, nn::Network{T,N}, x::AbstractVector{T}) where {T,N}
    empty!(s)
	ŷ = forward!(s, nn.layers[1], x)
	for i in 2:N-1
        ŷ .= forward!(s, nn.layers[i], ŷ)
	end
	forward!(s, nn.layers[N], ŷ)
end

function reverse_opt!(
        stacks::Vector{DT{T}}, 
        nn::Network{T,N}, 
        ŷs::Vector{<:AbstractVector{T}}, 
        ys::Vector{<:AbstractVector{T}}, 
        opts::Vector{Tuple{O,O}},
        normalize::Bool,
    ) where {T,N,O}
    any(isempty.(stacks)) && error("Stack is empty; did you run the forward pass?")
    Js = [∂C∂(ŷ, y) for (ŷ, y) in zip(ŷs, ys)]
    if normalize
        @debug "Jacobian norm" norm(Js[1])
        Js ./= [√norm(J) for J in Js]  # √x brings x halfway to 1 in log space
        @debug "Jacobian norm after scaling" norm(Js[1])
    end
    for (i, (l, (optW,optb))) in enumerate(zip(reverse(nn.layers), reverse(opts)))
        vals = pop!.(stacks)
        @debug "Jacobian norms layer $(length(layers)-i+1):" J=norm(Js[1]) ∂∂ξ=norm(∂∂ξ(l, vals[1][:bi]))
        ∇Wₗ = mean(tmap(zip(Js, vals)) do (J, (ξ, bi, _))
            J*LazyJac(ξ, bi)
        end)  # TODO: Mean accum takes a lot of time
        ΔW = apply!(optW, l.W, ∇Wₗ)
        @inbounds l.W .= l.W - ΔW
        ∇bₗ = mean(tmap(zip(Js, vals)) do (J, (ξ, bi, _))
            # TODO: Combine loops
            J'.*∂∂b(ξ, bi)
        end)
        @inbounds l.b .= l.b - apply!(optb, l.b, ∇bₗ)

        if i < N
            Js = [  # Skip jacobian computation on last layer
                J*∂∂ξ(l, bi)  # TODO: LazyJac?
                for (J, (_, bi, _)) in zip(Js, vals)
            ]
        end
    end
end

# opts = [(ADAM(), ADAM()) for l in layers]
function train_batch!(ds::Vector{DT{T}}, network, xs::Vector{<:AbstractVector{T}}, ys::Vector{<:AbstractVector{T}}, opts, normalize::Bool = false) where T
    ŷs = similar(ys)
    n = length(xs)
    @threads for i in 1:n  # TODO: Parallelize
        ŷs[i] = forward!(ds[i], network, xs[i])
    end
    reverse_opt!(ds, network, ŷs, ys, opts, normalize)
    mean(C(network(x), y) for (x,y) in zip(xs,ys))
end

function train_batch!(network, xs::Vector{<:AbstractVector{T}}, ys::Vector{<:AbstractVector{T}}, opts, normalize::Bool = false) where T
    ds = [DT{T}() for _ in 1:length(xs)]
    train_batch!(ds, network, xs, ys, opts, normalize)
end
