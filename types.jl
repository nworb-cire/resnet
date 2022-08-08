using DataStructures: Deque

if !@isdefined BoolVector
    if @isdefined MtlArray
        BoolVector = Union{
            BitVector,
            MtlArray{Bool, 1}
        }
    else
        BoolVector = BitVector
    end
end

if !@isdefined DT
    DT{T} = Deque{
        NamedTuple{(:ξ, :bi, :l), Tuple{
            Vector{T}, 
            BoolVector, 
            Base.RefValue{<:AbstractNetworkLayer{T}}
        }}
    } where T
end