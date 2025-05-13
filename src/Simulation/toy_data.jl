
function _generate_params(param, default_range, rng::AbstractRNG)
    if isnothing(param)
        rand(rng, Uniform(default_range...))
    elseif isa(param, Tuple)
        rand(rng, Uniform(param...))
    elseif isa(param, Vector)
        rand(rng, param)
    else
        param
    end
end

"""
```Julia
trendy_sine(T::Int, n::Int; period=nothing, slope=nothing, phase=nothing, sigma=0.0, 
    rng=Random.GLOBAL_RNG, return_metadata=true) -> Tuple{Matrix{Float64}, Dict{Symbol, Any}}
```
Generate `n` time series of length `T`, each composed of a sine wave with an optional linear trend and Gaussian noise defined by the
equation:
```math
x_t = \\sin\\left(\\frac{2\\pi}{\\tau}t + \\psi\\right) + \\frac{m t}{T} + \\sigma n_t
```
with period ``\\tau``, time point ``t``, linear trend slope ``m``, phase offset ``\\psi``, noise scale ``\\sigma`` and ``n_t \\sim \\mathcal{N}(0,1)``
# Arguments
- `T::Int`: Length of each time series
- `n::Int`: Number of time series instances to generate

# Keyword Arguments
- `period`: Period of the sinusoid, τ
    * `nothing`: Random values between 1.0 and 50.0 (default)
    * `Float64`: Fixed period for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (1.0, 20.0) → τ ~ U(1.0, 20.0)
    * `Vector`: Sample from discrete uniform distribution, e.g., τ ∈ 10, 20, 30
- `slope`: Linear trend gradient, m
    * `nothing`: Random values bewteen -5.0 and 5.0 (default)
    * `Float64`: Fixed slope for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (-3.0, 3.0) → m ~ U(-3.0, 3.0)
    * `Vector`: Sample from discrete uniform distribution, e.g., m ∈ -3.0, 0.0, 3.0
- `phase`: Phase offset, ψ
    * `nothing`: Random values between 0 and 2π (default)
    * `Float64`: Fixed phase for all time series
    * `Tuple`: Bounds for uniform random values, e.g., (0.0, π) → ψ ~ U(0.0, π)
    * `Vector`: Sample from discrete uniform distribution
- `sigma::Real`: Standard deviation of Gaussian noise, σ (default: 0.0)
- `rng::AbstractRNG`: Random number generator for reproducibility (default: Random.GLOBAL_RNG)
- `return_metadata::Bool`: Return generation parameters (default: true)

# Returns
- Matrix{Float64} of shape (n, T)
- Dictionary of generation parameters (:period, :slope, :phase, :sigma, :T, :n)
"""
function trendy_sine(T::Int, n::Int; period::Union{Nothing, Real, Tuple, Vector}=nothing, slope::Union{Nothing, Real, Tuple, Vector}=nothing, 
    phase::Union{Nothing, Real, Tuple, Vector}=nothing, sigma::Real=0.0, return_metadata::Bool=true, rng::AbstractRNG=Random.GLOBAL_RNG)

    # set default ranges for random
    DEFAULT_RANGES = (
        pe = (1.0, 50.0),
        sl = (-5.0, 5.0),
        ph = (0.0, 2π)
    )
    # Generate parameter vectors
    period_vals = [_generate_params(period, DEFAULT_RANGES.pe, rng) for _ in 1:n]
    slope_vals = [_generate_params(slope, DEFAULT_RANGES.sl, rng) for _ in 1:n]
    phase_vals = [_generate_params(phase, DEFAULT_RANGES.ph, rng) for _ in 1:n]
    
    X = Matrix{Float64}(undef, n, T)
    ts = 1:T
    for (i, series) in enumerate(eachrow(X))
        @. series = sin(2pi/period_vals[i] * ts + phase_vals[i]) + (slope_vals[i] * ts) / T + sigma * randn(rng)
    end

    info = nothing
    if return_metadata
        info = Dict(
            :period => period_vals,
            :slope => slope_vals,
            :phase => phase_vals,
            :sigma => sigma,
            :T => T,
            :n => n
        )
    end
    return X, info
end

function _single_state_space(T::Int; s::Int=2, sigma::Float64=0.3, rng::AbstractRNG=Random.GLOBAL_RNG)
    T += s # include burn-in
    xs = zeros(Float64, T)
    thetas = zeros(Float64, T)
    lambdas = zeros(Float64, T)
    mus = zeros(Float64, T)

    @inbounds for i in s:T
        theta = 0.0
        for j in 1:(s-1)
            theta += -thetas[i-j]
        end
        theta += sigma * randn(rng)
        lambda = lambdas[i-1] + sigma * randn(rng)
        mu = mus[i-1] + lambdas[i-1] + sigma * randn(rng)
        x = mu + theta + sigma * randn(rng)
        xs[i], mus[i], lambdas[i], thetas[i] = x, mu, lambda, theta
    end
    return xs[(s+1):end]
end

"""
    state_space(T::Int, n::Int, s::Int=2; sigma::Float64=0.3, rng::AbstractRNG}=Random.GLOBAL_RNG) -> Matrix{Float64}

Generate `n` time series of length `T` each from a state space model with residual terms drawn from a normal distribution
N(0, `sigma`) and lag order `s`. Time series are generated from the following model:
```math
x_t = \\mu_t + \\theta_t + \\eta_t
\\mu_t = \\mu_{t-1} + \\lambda_{t-1} + \\xi_t
\\lambda_t = \\lambda_{t-1} + \\zeta_{t}
\\theta_t = \\sum_{j=1}^{s-1} - \\theta_{t-j} + \\omega_t
```
where ``x_t`` is the ``t``-th value in the time series, and the residual terms ``\\eta_t``, ``\\xi_t``, ``\\zeta_t`` and ``\\omega_t`` are
randomly drawn from a normal distribution ``\\mathcal{N}(0, \\sigma)``.

# Arguments
- `T` -- Time series length.
- `n` -- Number of time-series instances.

# Keyword Arguments
- `s` -- Lag order (optional, default: `2`).
- `sigma` -- Noise standard deviation (optional, default: `0.3`).
- `rng` -- Random number generator of type `AbstractRNG` (optional, default: `Random.GLOBAL_RNG`).

# Returns 
A `Matrix{Float64}` of shape (n, T) containing the simulated time-series instances. 
"""
function state_space(T::Int, n::Int; s::Int=2, sigma::Float64=0.3, rng::AbstractRNG=Random.GLOBAL_RNG)
    if s < 2
        throw(ArgumentError("Lag order s must be ≥ 2."))
    end
    X = Matrix{Float64}(undef, n, T)
    for i in 1:n
        X[i, :] = _single_state_space(T; s=s, sigma=sigma, rng=rng)
    end
    return X
end
