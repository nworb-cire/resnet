using Test, SafeTestsets

begin
    @safetestset "Gradients" include("test_grads.jl")
    @safetestset "Feedforward" include("test_feedforward.jl")
    @safetestset "Residual" include("test_residual.jl")
end