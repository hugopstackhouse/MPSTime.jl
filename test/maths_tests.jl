# Value checks of the mathematics behind MPSTime, on problems small enough to compute exactly:
# orthonormality of the encodings, the loss and its gradient, normalisation of the learned density,
# and the conditional distributions used for imputation (against a brute-force dense wavefunction).
using MPSTime, Random, Statistics, Test
using MPSTime.ITensors, MPSTime.LinearAlgebra, MPSTime.QuadGK # dependencies of MPSTime, not of the test environment

const MT = MPSTime

# Gram matrix G_ij = ∫ conj(b_i(x)) b_j(x) dx over [a, b], with breakpoints where the basis has kinks
gram(f, a, b; pts=()) = quadgk(x -> (v = f(x); conj.(v) * transpose(v)), a, pts..., b; rtol=1e-8, atol=1e-10)[1]

# an encoder x -> b(x) for site ti, with any data-driven arguments fitted to X_norm (time series as columns)
function site_encoder(encoding, d; aux_basis_dim=2, projected_basis=false, X_norm=nothing, ti=1)
    opts = MT.safe_options(MPSOptions(; d=d, aux_basis_dim=aux_basis_dim, encoding=encoding, projected_basis=projected_basis, verbosity=-10))
    enc = opts.encoding
    X = isnothing(X_norm) ? zeros(1, 2) : X_norm # non-data-driven initialisers ignore the data
    args = enc.init(X, zeros(Int, size(X, 2)); opts=opts)
    f = enc.istimedependent ? (x -> enc.encode(x, d, ti, args...)) : (x -> enc.encode(x, d, args...))
    return f, enc.range, args
end

@testset "Encodings are orthonormal" begin
    # skewed (log-normal) and smooth (normal) data mapped to [-1, 1], time points as rows, 400 series as columns
    X_norm = let Z = exp.(randn(Xoshiro(42), 3, 400))
        2 .* (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z)) .- 1
    end
    X_smooth = let Z = 0.6 .* randn(Xoshiro(5), 4, 400)
        2 .* (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z)) .- 1
    end

    # orthonormal bases: G = I
    f, (a, b), _ = site_encoder(:Legendre, 6)
    @test gram(f, a, b) ≈ I atol=1e-8

    for ti in 1:2
        f, (a, b), _ = site_encoder(:Legendre, 4; projected_basis=true, X_norm=X_smooth, ti=ti)
        @test gram(f, a, b) ≈ I atol=1e-8
    end

    # split bases: each bin's auxiliary basis must be orthonormal on that bin
    f, (a, b), args = site_encoder(uniform_split(:legendre), 6; aux_basis_dim=3)
    bins = args[2][1]
    @test gram(f, a, b; pts=bins[2:end-1]) ≈ I atol=1e-8

    for ti in 1:3
        f, (a, b), args = site_encoder(histogram_split(:legendre), 6; aux_basis_dim=3, X_norm=X_norm, ti=ti)
        bins = args[2][1][ti]
        @test !allequal(diff(bins)) # the histogram bins have unequal widths for skewed data
        @test gram(f, a, b; pts=bins[2:end-1]) ≈ I atol=1e-8
    end

    f, (a, b), args = site_encoder(histogram_split(:uniform), 4; aux_basis_dim=1, X_norm=(X_norm .+ 1) ./ 2)
    bins = args[2][1][1]
    @test gram(x -> Float64.(f(x)), a, b; pts=bins[2:end-1]) ≈ I atol=1e-8

    # data-driven Sahand-Legendre bases (orthonormalised by trapezoidal quadrature on a 200-point grid, so only
    # approximately orthonormal, and less so for sharply peaked densities: use smooth data here)
    for enc in [:SLTD, :SL]
        f, (a, b), _ = site_encoder(enc, 4; X_norm=X_smooth)
        G = gram(f, a, b)
        @test all(isfinite, G)
        @test G ≈ I atol=1e-2
    end

    # bases that are orthogonal with a constant norm: G = c * I. A constant c rescales the density
    # by c^T but leaves the KLD gradients, the classifications, and the normalised conditionals unchanged
    for (enc, d) in [(:Legendre_Norm, 5), (:Fourier, 5), (:Stoudenmire, 2)]
        f, (a, b), _ = site_encoder(enc, d)
        G = gram(f, a, b)
        c = real(G[1, 1])
        @test c > 0
        @test G ≈ c * I atol=1e-8
    end
end


@testset "Conditional pdf matches a brute-force dense wavefunction" begin
    T, d, chi = 5, 3, 3
    rng = Xoshiro(7)
    Xtr, _ = trendy_sine(T, 120; period=(3.0, 6.0), slope=0.0, sigma=0.2, rng=rng)
    Xte, _ = trendy_sine(T, 4; period=(3.0, 6.0), slope=0.0, sigma=0.2, rng=rng)
    opts = MPSOptions(; d=d, chi_max=chi, nsweeps=2, eta=0.1, verbosity=-10, log_level=0, sigmoid_transform=false)
    tm, _, _ = fitMPS(Xtr, opts)
    imp = init_imputation_problem(tm, Xte; verbosity=-10, dx=1e-3)
    imp_dynamic = init_imputation_problem(tm, Xte; verbosity=-10, dx=1e-3, static_xvecs=false) # Vector instead of SVector states

    mps = imp.mpss[1]
    sites = MT.get_siteinds(mps)
    psi_full = ITensor(1.0)
    for t in mps
        psi_full *= t
    end
    psi = array(psi_full, sites...) # d^T dense amplitudes
    @test sum(abs2, psi) ≈ 1 atol=1e-10

    xvals = imp.x_guess_range.xvals
    enc(x, j) = MT.get_state(x, imp.opts, j, imp.enc_args)
    Phi = hcat([enc(x, 1) for x in xvals]...) # d x nx (time-independent encoding)
    trapz_cum(x, f) = vcat(0.0, cumsum((f[1:end-1] .+ f[2:end]) ./ 2 .* diff(x)))

    # condition psi on the observed values, then walk through the missing sites in order: at each site the
    # conditional pdf is phi(x)' * rho * phi(x) with rho the single-site reduced density matrix (PRR Eq. D1),
    # and the next site is conditioned on the value imputed here
    function brute_force(xs_scaled, missing_sites, imputed)
        A = copy(psi)
        for o in sort(setdiff(1:T, missing_sites); rev=true)
            v = conj(enc(xs_scaled[o], o))
            A = dropdims(sum(A .* reshape(v, ntuple(k -> k == o ? d : 1, ndims(A))), dims=o), dims=o)
        end
        medians, cdfs = Float64[], Vector{Float64}[]
        for m in sort(missing_sites)
            Am = reshape(A, d, :)
            rho = Am * Am'
            pdf = [real(dot(Phi[:, k], rho * Phi[:, k])) for k in axes(Phi, 2)]
            c = trapz_cum(xvals, pdf)
            c ./= c[end]
            push!(medians, xvals[argmin(abs.(c .- 0.5))])
            push!(cdfs, c)
            v = conj(enc(imputed[m], m))
            A = reshape(transpose(v) * Am, size(A)[2:end]...)
        end
        return medians, cdfs
    end

    _, norms = MT.transform_train_data(permutedims(imp.X_train); opts=imp.opts)
    for (inst, missing_sites) in [(1, [2, 3, 4]), (2, [1, 2, 3, 4]), (3, [3, 4, 5]), (4, [1, 3, 5])]
        ts, _, _, _, _ = MPS_impute(imp, 0, inst, missing_sites, :median; invert_transform=false, NN_baseline=false, plot_fits=false, get_metrics=false)
        cdfs, _, _, _ = get_cdfs(imp, 0, inst, missing_sites)

        # the observed values, scaled exactly as get_predictions does
        xs = copy(Xte[inst, :])
        xs[missing_sites] .= mean(imp.X_train)
        xs_scaled, _ = MT.transform_test_data(xs, norms; opts=imp.opts)

        medians, cdfs_ref = brute_force(xs_scaled, missing_sites, ts[1])
        @test ts[1][sort(missing_sites)] ≈ medians atol=1e-12
        for k in eachindex(cdfs_ref)
            @test cdfs[k] ≈ cdfs_ref[k] atol=1e-8
        end

        ts_dynamic, _, _, _, _ = MPS_impute(imp_dynamic, 0, inst, missing_sites, :median; invert_transform=false, NN_baseline=false, plot_fits=false, get_metrics=false)
        @test ts_dynamic[1] ≈ ts[1] atol=1e-12
    end
end
