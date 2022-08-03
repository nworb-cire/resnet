using Test, SafeTestsets

begin
    @safetestset "Gradients" include("test_grads.jl")
    @safetestset "Backprop" include("test_backprop.jl")
end