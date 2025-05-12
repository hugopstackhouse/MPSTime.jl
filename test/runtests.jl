using MPSTime
using Test
using TestItems
using Aqua
using Random

@testset "Aqua.jl Quality Assurance" begin
    Aqua.test_ambiguities(MPSTime) # test for method ambiguities
    Aqua.test_unbound_args(MPSTime) # test that all methods have bounded type parameters
    Aqua.test_undefined_exports(MPSTime) # test that all exported names exist
    Aqua.test_stale_deps(MPSTime) # test that the package loads all deps listed in the root Project.toml
    Aqua.test_piracies(MPSTime) # test that the package does not commit type piracies
end

@testset "Bases" begin 
    include("basis_tests.jl")
end

@testset "Save/Load TrainedMPS" begin 
    include("save_load.jl")
end

@testset "Analysis" begin
    include("analysis_tests.jl")
end

@testset "Classifier" begin
    include("classification.jl")
end

@testset "Imputation" begin
    include("imputation.jl")
end

@testset "Imputation Data Utils" begin
    include("simulation_tests.jl")
end
