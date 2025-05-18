using Statistics

# generate some random data for the tests
test_data = rand(5, 10_000);

@testset "mcar" begin 
    # test the missing completely at random mechanism
    pm = 0.3 # 30% missing
    outX, outInd = mcar(test_data[1, :], pm)
    @test isa(outX, Vector)
    @test !isnothing(outInd)
    @test isa(outInd, Vector{Int})
    # check percentage missing expectation
    ntrials = 10_000
    nmissing = zeros(Int64, ntrials)
    for trial in 1:ntrials
        outX, outInd = mcar(test_data[1, :], pm)
        nmissing[trial] = length(outInd)
    end
    # check for more than 1% difference
    @test isapprox(mean(nmissing), (pm * size(test_data, 2)); rtol=0.01)
    # check for reproducibility/randomness
    rng1 = Xoshiro(123);
    outX1, outInd1 = mcar(test_data[1, :], pm; rng=rng1)
    rng2 = Xoshiro(123)
    outX2, outInd2 = mcar(test_data[1, :], pm; rng=rng2)
    @test isequal(outX1, outX2)
    @test isequal(outInd1, outInd2)
    @test !isequal(outX1, outX)
    @test !isequal(outInd1, outInd)

    # invalid fraction missing
    @test_throws ArgumentError mcar(test_data[1, :], 1.1)
    @test_throws ArgumentError mcar(test_data[1, :], -0.1)
end

@testset "mar" begin
    pm = 0.3 # 30% missing
    outX, outInd = mar(test_data[1, :], pm)
    @test !isnothing(outX)
    @test isa(outX, Vector)
    @test !isnothing(outInd)
    @test isa(outInd, Vector{Int})
    
    # check percentage missing
    @test isequal(length(outInd), (pm * size(test_data, 2)))
    
    # check for reproducibility/randomness
    rng1 = Xoshiro(123);
    rng2 = Xoshiro(123);
    outX1, outInd1 = mar(test_data[1, :], pm; rng=rng1)
    outX2, outInd2 = mar(test_data[1, :], pm; rng=rng2)
    @test isequal(outX1, outX2)
    @test isequal(outInd1, outInd2)
    @test !isequal(outX1, outX)
    @test !isequal(outInd1, outInd)

    # invalid fraction missing
    @test_throws ArgumentError mar(test_data[1, :], 1.1)
    @test_throws ArgumentError mar(test_data[1, :], -0.1)
    
end

@testset "mnar" begin 
    pm = 0.3 # 30% missing
    outX_lowest, outInd_lowest = mnar(test_data[1, :], pm) # default is lowest
    outX_highest, outInd_highest = mnar(test_data[1, :], pm, MPSTime.HighestMNAR()) # default is lowest
    @test !isnothing(outX_lowest)
    @test !isnothing(outX_highest)
    @test isa(outX_lowest, Vector)
    @test isa(outX_highest, Vector)
    @test !isnothing(outInd_lowest)
    @test !isnothing(outInd_highest)
    @test isa(outInd_lowest, Vector{Int})
    @test isa(outInd_highest, Vector{Int})

    # check percentage missing
    @test isequal(length(outInd_lowest), (pm * size(test_data, 2)))
    @test isequal(length(outInd_highest), (pm * size(test_data, 2)))

    # invalid fraction missing
    @test_throws ArgumentError mnar(test_data[1, :], 1.1)
    @test_throws ArgumentError mnar(test_data[1, :], -0.1)

end

@testset "trendy sinusoid" begin
    # basic functionality tests
    T = 100;
    n = 10;
    X, info = trendy_sine(T, n)
    @test isa(X, Matrix)
    @test size(X) == (n, T)
    @test isa(info, Dict)
    X, info_nometa = trendy_sine(T, n; return_metadata=false)
    @test isnothing(info_nometa)
    # check reproducibility
    rng1 = Xoshiro(1234);
    X1, _ = trendy_sine(T, n; rng=rng1)
    rng2 = Xoshiro(1234);
    X2, _ = trendy_sine(T, n; rng=rng2)
    @test isequal(X1, X2)
    # parameter checks
    period_fixed = 10.0
    slope_fixed = 3.0
    phase_fixed = pi

    period_cont = (10, 30)
    slope_cont = (-3.0, 3.0)
    phase_cont = (0, pi)

    period_discr = [10, 20, 30];
    slope_discr = [-3, 0, 3]
    phase_discr = [0, pi/2, pi, 3pi/2]

    _, info_fixed = trendy_sine(T, n; period=period_fixed, phase=phase_fixed, slope=slope_fixed, return_metadata=true)
    _, info_cont = trendy_sine(T, n; period=period_cont, phase=phase_cont, slope=slope_cont, return_metadata=true)
    _, info_disc = trendy_sine(T, n; period=period_discr, phase=phase_discr, slope=slope_discr, return_metadata=true)
    @test all(x -> x .== period_fixed, info_fixed[:period])
    @test all(x -> first(period_cont) <= x <= last(period_cont), info_cont[:period])
    @test all([x in period_discr for x in info_disc[:period]])

    @test all(x -> x .== slope_fixed, info_fixed[:slope])
    @test all(x -> first(slope_cont) <= x <= last(slope_cont), info_cont[:slope])
    @test all([x in slope_discr for x in info_disc[:slope]])

    @test all(x -> x .== slope_fixed, info_fixed[:slope])
    @test all(x -> first(slope_cont) <= x <= last(phase_cont), info_cont[:phase])
    @test all([x in phase_discr for x in info_disc[:phase]])

    # test defaults
    X, info_default = trendy_sine(100, 10; return_metadata=true)
    @test all(x -> -5.0 <= x <= 5.0, info_default[:slope])
    @test all(x -> 1.0 <= x <= 50.0, info_default[:period])
    @test all(x -> 0.0 <= x <= 2pi, info_default[:phase])

end

@testset "state space model" begin
    # basic functionality tests
    T = 100;
    n = 10;
    X = MPSTime.state_space(T, n);
    @test isa(X, Matrix)
    @test size(X) == (n, T)
    # check reproducibility
    rng1 = Xoshiro(745)
    X1 = MPSTime.state_space(T, n; rng=rng1)
    rng2 = Xoshiro(745)
    X2 =  MPSTime.state_space(T, n; rng=rng2)
    @test isequal(X1, X2)
    @test !isequal(X, X1)
    # check errors
    @test_throws ArgumentError MPSTime.state_space(T, n; s=1)
    @inferred MPSTime.state_space(T, n);
end
