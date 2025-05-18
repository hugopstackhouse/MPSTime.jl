function get_state(
        x::Float64, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector
    )
    """Get the state for a time dependent encoding at site j"""
    if opts.encoding.istimedependent
        # enc_args_concrete = convert(Vector{Vector{Vector{Int}}}, enc_args) # https://i.imgur.com/cmFIJmS.png
        state = opts.encoding.encode(x, opts.d, j, enc_args...)
    else
        state = opts.encoding.encode(x, opts.d, enc_args...)
    end

    return state
    
end

function get_conditional_probability(
    state::AbstractVector, 
    A::Matrix, 
)
    p = BLAS.gemv('C', A, state)
    return abs(dot(p,p)) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end

function get_conditional_probability(
    state::AbstractVector, 
    A::Vector, 
)
    # p = BLAS.gemv('C', A, state)
    return abs2(dot(state,A)) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end

function get_conditional_probability(
    state::SVector, 
    A::Matrix, 
)
    p = state'* A
    return abs(dot(p,p)) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end

function get_conditional_probability(
    state::SVector{D}, 
    rdm::MMatrix{D,D}, 
) where D
    return abs(dot(state, rdm, state)) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end


function get_normalisation_constant(xs::AbstractVector{Float64}, pdf::AbstractVector{Float64})
    return NumericalIntegration.integrate(xs, pdf, NumericalIntegration.TrapezoidalEvenFast())

end





function get_mean_from_rdm(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Union{Float64, Nothing},
        dx::Float64;
        get_std::Bool=true
    )

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        @inline prob = get_conditional_probability(state, rdm) 
        probs[index] = prob
    end

    Z = get_normalisation_constant(samp_xs, probs)

    # expectation
    expect_x = mapreduce(*,+,samp_xs, probs) * dx / Z
    expect_state = get_state(expect_x, opts, j, enc_args) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

    std_val = 0.
    if get_std
        # variance
        squared_diffs = (samp_xs .- expect_x).^2
        var = mapreduce(*,+,squared_diffs, probs) * dx / Z

        std_val = sqrt(var)

    end

    return expect_x, expect_state, std_val

end


function get_mode_from_rdm(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        x_prev::Union{Float64, Nothing}, 
        max_jump::Union{Number, Nothing}=nothing
    )
    """Much simpler approach to get the mode of the conditional 
    pdf (cpdf) for a given rdm. Simply evaluate P(x) over the x range,
    with interval dx, and take the argmax."""
    # don't even need to normalise since we just want the peak
    # Z = get_normalisation_constant(s, rdm, opts, j, enc_args)

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        @inline prob = get_conditional_probability(state, rdm)
        probs[index] = prob
    end
    
    # get the mode of the pdf
    if isnothing(x_prev) || isnothing(max_jump)
        mode_idx = argmax(probs)
    else
        perm = sortperm(probs;rev=true)
        mode_idx = 0
        for i in perm
            if abs(samp_xs[i] - x_prev) <= max_jump
                mode_idx = i
                break
            end
        end
        if mode_idx == 0
            @warn("No valid guess withing max_jump of the previous imputation point. Increase max_jump")
            mode_idx = perm[1]
        end
    end
    mode_x = samp_xs[mode_idx]
    mode_state = samp_states[mode_idx]

    return mode_x, mode_state, 0. # the mode has no way to compute the error, hence always returning 0

end

get_mode_from_rdm(
    rdm::AbstractVecOrMat, 
    samp_xs::AbstractVector{Float64}, 
    samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
    s::Index, 
    opts::Options,
    j::Integer,
    enc_args::AbstractVector,
    x_prev::Union{Number, Nothing}=nothing, 
    max_jump::Union{Number, Nothing}=nothing
) = get_mode_from_rdm(rdm, samp_xs, samp_states, s, x_prev, max_jump)



function get_median_from_rdm(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Union{Float64, Nothing};
        get_wmad::Bool=true
    )
    # return the median and the weighted median absolute deviation as a measure of uncertainty 

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        @inline prob = get_conditional_probability(state, rdm)
        probs[index] = prob
    end

    cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
    Z = cdf[end]
    cdf /= Z
    probs /= Z

    median_arg = argmin(@. abs(cdf - 0.5))

    median_x = samp_xs[median_arg]
    median_s = samp_states[median_arg] / sqrt(Z)
    # median_s = itensor(get_state(median_x, opts, j, enc_args), s) / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

    wmad_x = 0.
    if get_wmad
        # get the weighted median abs deviation
        wmad_x = median(abs.(samp_xs .- median_x), pweights(probs))
    end
    return (median_x, median_s, wmad_x)

end





function get_median_and_cdf(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Union{Float64, Nothing};
        get_wmad::Bool=true
    )
    # return the median and the weighted median absolute deviation as a measure of uncertainty 

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        @inline prob = get_conditional_probability(state, rdm)
        probs[index] = prob
    end

    cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
    Z = cdf[end]
    cdf /= Z
    probs /= Z

    median_arg = argmin(@. abs(cdf - 0.5))

    median_x = samp_xs[median_arg]
    median_s = samp_states[median_arg] / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 

    wmad_x = 0.
    if get_wmad
        # get the weighted median abs deviation
        wmad_x = median(abs.(samp_xs .- median_x), pweights(probs))
    end
    return (median_x, median_s, wmad_x, cdf)

end


function get_cdf(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}},
        s::Index
    )

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        @inline prob = get_conditional_probability(state, rdm)
        probs[index] = prob
    end

    cdf = NumericalIntegration.cumul_integrate(samp_xs, probs, NumericalIntegration.TrapezoidalEvenFast())
    Z = cdf[end]
    cdf /= Z
    return cdf
end

function get_sample_from_rdm(
        rdm::AbstractVecOrMat, 
        samp_xs::AbstractVector{Float64}, 
        samp_states::AbstractVector{<:AbstractVector{<:Number}}, 
        s::Index, 
        opts::Options, 
        j::Integer,
        enc_args::AbstractVector,
        x_prev::Union{Float64, Nothing};
        rng::AbstractRNG,
        rejection_threshold::Union{Float64, Symbol}, 
        max_trials::Integer,
    )
    """Sample from the conditional distribution defined by the rdm, but 
    reject samples if they exceed a predetermined threshold which is set
    by the weighted median absolute deviation (WMAD).
    - threshold is the multiplier for WMAD as threshold i.e., threshold*WMAD 
    - atol is abs tolerance for the root finder
    - max trials is the maximum number of rejections
    """
    # sample without rejection if threshold is none
    sampled_x = 0.
    if rejection_threshold == :none
        u = rand(rng)
        #cdf_wrapper(x) = get_cdf(x, rdm, Z, opts, enc_args) - u
        cdf = get_cdf(rdm, samp_xs, samp_states, s)
        Z = cdf[end]
        x_ind = argmin(@. abs(cdf/Z - u ) )
        sampled_x = samp_xs[x_ind]
        wmad = 0.
        # sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
    else
        # now determine the median and wmad - don't need high precision here, just a general ballpark
        median, _, wmad, cdf = get_median_and_cdf(rdm, samp_xs, samp_states, s, opts, j, enc_args, x_prev; get_wmad=true)
        Z = cdf[end]
        rejections = 0 # rejected :(
        for i in 1:max_trials
            u = rand(rng) # sample a random value from ~ U(0, 1)
            # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
            x_ind = argmin(@. abs(cdf/Z - u ) )
            sampled_x = samp_xs[x_ind]
            # sampled_x = find_zero(x -> get_cdf(x, rdm, Z, opts, j, enc_args) - u, opts.encoding.range; atol=atol)
            # choose whether to accept or reject
            if abs(sampled_x - median) < rejection_threshold*wmad
                break 
            end
            rejections += 1
        end
        #@show rejections
    end
    # map sampled x_k back to a state
    sampled_state = samp_states[x_ind] / sqrt(Z) # the 1/sqrt(Z) fixes the normalisation when reconditioning 
    return sampled_x, sampled_state, wmad 
end


function compute_entanglement_entropy_profile(class_mps::MPS)
    """Compute the entanglement entropy profile (page curve)
    for an un-labelled (class) mps.
        """
    mps = deepcopy(class_mps)
    @assert isapprox(norm(mps), 1.0; atol=1e-3) "MPS is not normalised!"
    mps_length = length(mps)
    entropy_vals = Vector{Float64}(undef, mps_length)
    # for each bi-partition coordinate, compute the entanglement entropy
    for oc in eachindex(mps)
        orthogonalize!(mps, oc) # shift orthogonality center to bipartition coordinate
        if oc == 1 || oc == mps_length
            _, S, _ = svd(mps[oc], (siteind(mps, oc)))
        else
            _, S, _ = svd(mps[oc], (linkind(mps, oc-1), siteind(mps, oc)))
        end
        SvN = 0.0
        # loop over the diagonal of the singular value matrix and extract the values
        for n = 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            if (p > 1E-12) # to avoid log(0)
                SvN += -p * log(p)
            end
        end
        entropy_vals[oc] = SvN
    end

    return entropy_vals

end
