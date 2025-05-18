using JLD2

# load data
f = JLD2.jldopen("Data/ecg200/datasets/ecg200.jld2", "r");
X_train = read(f, "X_train");
X_test = read(f, "X_test");
y_train = read(f, "y_train");
y_test = read(f, "y_test");

# train MPS on testing data
opts = MPSOptions(d=4, chi_max=25;nsweeps=1, log_level=-5, verbosity=-10)
mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts);

"""Basic functionality tests"""
# test the von_neumann_entropy function 
class_mpss, _ = MPSTime.expand_label_index(mps.mps)

@testset "von neumann entropy" begin
    class1_output = MPSTime.von_neumann_entropy(class_mpss[1])
    class2_output = MPSTime.von_neumann_entropy(class_mpss[2])
    @test !isnothing(class1_output)
    @test isa(class1_output, Vector)
    @test !isnothing(class2_output)
    @test isa(class2_output, Vector)
    @test !isnothing(MPSTime.von_neumann_entropy(class_mpss[1], log))
    @test !isnothing(MPSTime.von_neumann_entropy(class_mpss[1], log2)) 
    @test !isnothing(MPSTime.von_neumann_entropy(class_mpss[1], log10))
    @test_throws ArgumentError MPSTime.von_neumann_entropy(class_mpss[1], exp)
end

@testset "bipartite entropy" begin
    # test basic functioning of bipartite spectrum function
    bipartite_spectrum_output = MPSTime.bipartite_spectrum(mps);
    @test bipartite_spectrum_output !== nothing
    @test isequal(length(bipartite_spectrum_output), 2)
    # type checks
    @test isa(bipartite_spectrum_output, Vector{Vector{Float64}})
    @test isequal(length(bipartite_spectrum_output[1]), size(X_train, 2))
    @test isequal(length(bipartite_spectrum_output[2]), size(X_train, 2))
    # error checks
    @test_throws ArgumentError MPSTime.bipartite_spectrum(mps; logfn=exp)
end

@testset "one site rdm" begin
    # test basic functioning of one site rdm function
    test_site1, test_site2 = 2, 5
    rdm_out1 = MPSTime.one_site_rdm(class_mpss[1], test_site1);
    rdm_out2 = MPSTime.one_site_rdm(class_mpss[2], test_site2);
    # type checks
    @test !isnothing(rdm_out1)
    @test !isnothing(rdm_out2)
    @test isa(rdm_out1, Matrix)
    @test isa(rdm_out2, Matrix)
    # check properties of the rdm
    @test isequal(size(rdm_out1), (opts.d, opts.d)) # check the shape is (d, d)
    @test isequal(size(rdm_out2), (opts.d, opts.d))
end

@testset "single site entropy" begin
    # test basic functioning of single site entropy function
    sse_out1 = MPSTime.single_site_entropy(class_mpss[1])
    sse_out2 = MPSTime.single_site_entropy(class_mpss[2])
    @test !isnothing(sse_out1)
    @test !isnothing(sse_out2)
    # type checks
    @test isa(sse_out1, Vector)
    @test isequal(length(sse_out1), length(class_mpss[1]))
    @test isa(sse_out2, Vector)
    @test isequal(length(sse_out2), length(class_mpss[2]))
    # check all positive
    @test all(x -> x .>=0, sse_out1)
    @test all(x -> x .>=0, sse_out2)
end

@testset "single site spectrum" begin
    ss_spec = single_site_spectrum(mps);
    @test !isnothing(ss_spec)
    @test isequal(length(ss_spec), 2)
    @test isa(ss_spec, Vector{Vector{Float64}})
    @test isequal(length(ss_spec[1]), length(class_mpss[1]))
    @test isequal(length(ss_spec[2]), length(class_mpss[2]))
    @test all(x -> x .>=0, ss_spec[1])
    @test all(x -> x .>=0, ss_spec[2])
end

@testset "rdm correction" begin
    # test basic functioning of reduced density matrix checks
    rho = MPSTime.one_site_rdm(class_mpss[1], 2)
    rho_correct_output = MPSTime.rho_correct(rho)
    @test !isnothing(rho_correct_output)
    @test isa(rho_correct_output, Matrix)
    # expect same output as no correction required
    @test isequal(rho_correct_output, rho)
    # construct a toy rdm with negative eigenvalues outside tolerance
    toy_rdm_oot = [0.1 -1.03; -0.1 0.9]
    @test_throws DomainError MPSTime.rho_correct(toy_rdm_oot)
    # toy rdm with large negative eigenvalue inside tolerance
    @test_throws DomainError MPSTime.rho_correct(toy_rdm_oot, 1.0)
    # toy rdm with small negative eigenvalue inside tolerance
    toy_rdm_iot = [1.0 0; 0 -1e-10]
    corrected_toy_iot = MPSTime.rho_correct(toy_rdm_iot)
    @test !isnothing(corrected_toy_iot)
end


