include("grads.jl")

DT{T} = Deque{@NamedTuple{
	ξ::Vector{T},
	bi::BitVector,
	W::Base.RefValue{Matrix{S}}
}} where {T,S}

function forward!(s::DT{T,S}, ls::Vector{Layer{T}}, x::Vector{T}) where {S,T}
    empty!(s)
	ŷ = x
	for l in ls
		ỹ = muladd(l.W',ŷ,l.b)
		bi = !l.activation .| (ỹ .> 0)
		push!(s, (ξ=ŷ,bi=bi,W=Ref(l.W)))
		ŷ = ifelse(l.activation, relu.(ỹ), ỹ)
	end
	ŷ
end

function grads(s::DT{T,S}, layers::Vector{Layer{T}}, ŷ::Vector{T}, y::Vector{T}) where {S,T}
    isempty(s) && error("Stack is empty; did you run the forward pass?")
    ŷ, y
	J = ∂C∂(ŷ, y)
	∇W = Matrix{T}[]
	∇b = Vector{T}[]
	while !isempty(s)
		ξ, bi, refW = pop!(s)
		W = refW[]
		# TODO: ∂∂W is low-rank. Compute efficiently.
		∇Wₗ = reshape((J*LazyJac(ξ, bi))', size(W))
		# TODO: ∂∂b is sparse. Compute efficiently.
		∇bₗ = J'.*∂∂b(ξ, bi)
		push!(∇W,∇Wₗ)
		push!(∇b,∇bₗ)
		if !isempty(s)
			J = J*∂∂ξ(W, bi)  # Skip jacobian computation on last layer
		end
	end
	return zip(∇W, ∇b)
end
reverse! = grads

function reverse_opt!(stacks::Vector{DT{T,S}}, layers::Vector{Layer{T}}, ŷs::Vector{Vector{T}}, ys::Vector{Vector{T}}, opts::Vector{Tuple{O,O}}) where {S,T,O}
    any(isempty.(stacks)) && error("Stack is empty; did you run the forward pass?")
    Js = [∂C∂(ŷ, y) for (ŷ, y) in zip(ŷs, ys)]
    for (l, (optW,optb)) in zip(reverse(layers), reverse(opts))
        vals = pop!.(stacks)
        ∇Wₗ = mean([
            J*LazyJac(ξ, bi)
            for (J, (ξ, bi, _)) in zip(Js, vals)
        ])
        update!(optW, l.W, ∇Wₗ')
        ∇bₗ = mean([  # TODO: Combine for loops
            J'.*∂∂b(ξ, bi)
            for (J, (ξ, bi, _)) in zip(Js, vals)
        ])
        update!(optb, l.b, ∇bₗ)
		if !isempty(first(stacks))
			Js = [  # Skip jacobian computation on last layer
                J*∂∂ξ(refW[], bi)
                for (J, (_, bi, refW)) in zip(Js, vals)
            ]
		end
    end
end

# opts = [(ADAM(), ADAM()) for l in layers]
function train!(layers, xs, ys, opts)
    ŷs = similar(ys)
    n = length(xs)
    ds = [DT{Float32,Float32}() for _ in 1:n]
    for i in 1:n
        ŷs[i] = forward!(ds[i], layers, xs[i])
    end
    reverse_opt!(ds, layers, ŷs, ys, opts)
    mean(C(layers(x), y) for (x,y) in zip(xs,ys))
end
