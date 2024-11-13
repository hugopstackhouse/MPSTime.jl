import Base.*
contractTensor = ITensors._contract
*(t1::Tensor, t2::Tensor) = contractTensor(t1, t2)

include("./samplingUtils.jl");


mutable struct forecastable
    mps::MPS
    class::Int
    test_samples::Matrix{Float64}
    opts::Options
    enc_args::Vector{Any}
end

function condition_until_next!(
        i::Int,
        it::ITensor,
        x_samps::AbstractVector{Float64},
        known_sites::AbstractVector{Int},
        class_mps::MPS,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS
    )

    while i in known_sites
        # save the x we know
        known_x = timeseries[i]
        x_samps[i] = known_x

        # make projective measurement by contracting with the site
        it *= (class_mps[i] * dag(timeseries_enc[i]))
        i += 1
    end

    return i, it
end



function allocate_mem(dtype, j, sites, links, num_imputation_sites)
    if j == 1 
        inds = (sites[j], links[j])
    elseif j == num_imputation_sites
        inds = (sites[j], links[j-1])
    else
        inds = (sites[j], links[j-1], links[j])
    end

    dims = ITensors.dim.(inds)
    return ITensor(dtype, zeros(dtype, dims), inds)
    
end



function precondition(
        class_mps::MPS,
        opts::Options,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS,
        imputation_sites::AbstractVector{Int};
        mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
        dx::Float64=1E-4, 
        xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
    )
    s = siteinds(class_mps)
    known_sites = setdiff(collect(1:length(class_mps)), imputation_sites)
    total_known_sites = length(class_mps)
    total_impute_sites = length(imputation_sites)

    x_samps = Vector{Float64}(undef, total_known_sites) # store imputed samples
    original_mps_length = length(class_mps)

    mps_conditioned = Vector{ITensor}(undef, total_impute_sites)

    mps_cond_idx = 1
    # condition the mps on the known values
    i = 1
    while i <= original_mps_length
        if mps_cond_idx == total_impute_sites
            # at the last site, condition all remaining
            it = ITensor(1)
            if i in known_sites
                # condition all the known sites up until the last imputed site 
                i, it = condition_until_next!(i, it, x_samps, known_sites, class_mps, timeseries, timeseries_enc)
            end
            last_site = class_mps[i] # last imputed sites
            it2 = ITensor(1)
            i += 1
            # condition all the remaining sites in the mps (ok if there aren't any)
            i, it2 = condition_until_next!(i, it2, x_samps, known_sites, class_mps, timeseries, timeseries_enc)

            mps_conditioned[mps_cond_idx] = it * last_site * it2# normalize!(it * last_site * it2)

        elseif i in known_sites
            it = ITensor(1)
            i, it = condition_until_next!(i, it, x_samps, known_sites, class_mps, timeseries, timeseries_enc)
            mps_conditioned[mps_cond_idx] = it * class_mps[i] # normalize!(it * class_mps[i])
        else
            mps_conditioned[mps_cond_idx] = deepcopy(class_mps[i])
        end
        mps_cond_idx += 1
        i += 1
    end
    return x_samps, MPS(mps_conditioned)

end


function impute_at2!(
    mps::MPS,
    x_samps::AbstractVector{Float64},
    method::Function,
    opts::Options,
    enc_args::AbstractVector,
    imputation_sites::Vector{Int};
    mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
    dx::Float64=1E-4, 
    xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
    mode_index=Index(opts.d),
    xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
    xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
    kwargs...
)
inds = 1:length(mps)

orthogonalize!(mps, first(inds)) #TODO: this line is what breaks imputations of non sequential sites, fix


s = siteinds(mps)
lds = linkdims(mps)
lis = linkinds(mps)
errs = zeros(Float64, length(x_samps))#Vector{Float64}(undef, total_known_sites)
total_impute_sites = length(imputation_sites)


A = Array{opts.dtype}(undef, (opts.d, maximum(linkdims(mps))))
ld = lds[1]
A[:,1:ld] .= array(mps[first(inds)])

rdm = Array{opts.dtype}(undef, (opts.d, opts.d))

for (ii,i) in enumerate(inds)
    imp_idx = imputation_sites[i]
    site_ind = s[i]
    ld = (i == total_impute_sites) ? 1 : lds[i]


    rdm .= @views A[:,1:ld] * A[:, 1:ld]'

    mx, ms, err = method(itensor(rdm, (site_ind', site_ind)), xvals, xvals_enc, site_ind, opts, enc_args; kwargs...)
    x_samps[imp_idx] = mx
    errs[imp_idx] = err
   
    # recondition the MPS based on the prediction
    if ii != total_impute_sites
        ld2 = (i == total_impute_sites - 1) ? 1 : lds[i+1]
        @view(A[:, 1:ld2]) .= array(normalize!(mps[i + 1] * (dag(itensor(ms, site_ind); allow_alias=true) * itensor(A[:,1:ld], (s[i], lis[i])))))
        # A = normalize!(mps[inds[ii+1]] * Am)
        
        # A .= normalize!(A_new)
    end
end 
return (x_samps, errs)
end

function impute_at!(
        mps::MPS,
        x_samps::AbstractVector{Float64},
        method::Function,
        opts::Options,
        enc_args::AbstractVector,
        imputation_sites::Vector{Int};
        mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
        dx::Float64=1E-4, 
        xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        kwargs...
    )
    mps_inds = 1:length(mps)
    s = siteinds(mps)
    errs = zeros(Float64, length(x_samps))#Vector{Float64}(undef, total_known_sites)
    total_impute_sites = length(imputation_sites)


    orthogonalize!(mps, first(mps_inds)) #TODO: this line is what breaks imputations of non sequential sites, fix
    A = mps[first(mps_inds)]
    for (ii,i) in enumerate(mps_inds)
        imp_idx = imputation_sites[i]
        site_ind = s[i]

        rdm = prime(A, site_ind) * dag(A)

        mx, ms, err = method(rdm, xvals, xvals_enc, site_ind, opts, enc_args; kwargs...)
        x_samps[imp_idx] = mx
        errs[imp_idx] = err
       
        # recondition the MPS based on the prediction
        if ii != total_impute_sites
            ms = itensor(ms, site_ind)
            Am = A * dag(ms)
            # A = normalize!(mps[mps_inds[ii+1]] * Am)
            A = mps[mps_inds[ii+1]] * Am
            # A .= normalize!(A_new)
        end
    end 
    return (x_samps, errs)
end

"""
impute missing data points using the median of the conditional distribution (single site rdm ρ).

# Arguments
- `class_mps::MPS`: 
- `opts::Options`: MPS parameters.
- `timeseries::AbstractVector{<:Number}`: The input time series data that will be imputed.
- `timeseries_enc::MPS`: The encoded version of the time series represented as a product state. 
- `imputation_sites::Vector{Int}`: Indices in the time series where imputation is to be performed.
- `get_wmad::Bool`: Whether to compute the weighted median absolute deviation (WMAD) during imputation (default is `false`).

# Returns
A tuple containing:
- `median_values::Vector{Float64}`: The imputed median values at the specified imputation sites.
- `wmad_value::Union{Nothing, Float64}`: The weighted median absolute deviation if `get_wmad` is true; otherwise, `nothing`.

"""
function any_impute_directMedianOpt(
        class_mps::MPS,
        opts::Options,
        enc_args::AbstractVector,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS,
        imputation_sites::Vector{Int};
        mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
        dx::Float64=1E-4, 
        xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        get_wmad::Bool=true
    )
    
    x_samps, mps_conditioned = precondition(
        class_mps,
        opts, 
        timeseries,
        timeseries_enc,
        imputation_sites;
        mode_range=mode_range,
        dx=dx,
        xvals=xvals,
        mode_index=mode_index,
        xvals_enc=xvals_enc,
        xvals_enc_it=xvals_enc_it,
    )

    x_samps, x_wmads = impute_at!(
        mps_conditioned,
        x_samps,
        get_median_from_rdm,
        opts,
        enc_args,
        imputation_sites;
        mode_range=mode_range, 
        dx=dx ,
        xvals=xvals,
        mode_index=mode_index,
        xvals_enc=xvals_enc,
        xvals_enc_it=xvals_enc_it,
        get_wmad=get_wmad
    )

    return (x_samps, x_wmads)
end


"""
impute missing data points using the median of the conditional distribution (single site rdm ρ).

# Arguments
- `class_mps::MPS`: 
- `opts::Options`: MPS parameters.
- `timeseries::AbstractVector{<:Number}`: The input time series data that will be imputed.
- `timeseries_enc::MPS`: The encoded version of the time series represented as a product state. 
- `imputation_sites::Vector{Int}`: Indices in the time series where imputation is to be performed.
- `get_wmad::Bool`: Whether to compute the weighted median absolute deviation (WMAD) during imputation (default is `false`).

# Returns
A tuple containing:
- `median_values::Vector{Float64}`: The imputed median values at the specified imputation sites.
- `wmad_value::Union{Nothing, Float64}`: The weighted median absolute deviation if `get_wmad` is true; otherwise, `nothing`.

"""
function any_impute_directMedian(
        class_mps::MPS,
        opts::Options,
        enc_args::AbstractVector,
        timeseries::AbstractVector{<:Number},
        timeseries_enc::MPS,
        imputation_sites::Vector{Int};
        mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
        dx::Float64=1E-4, 
        xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        get_wmad::Bool=true
    )

    if isempty(imputation_sites)
        throw(ArgumentError("imputation_sites can't be empty!")) 
    end
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    total_known_sites = length(mps)
    num_imputation_sites = length(imputation_sites)
    x_samps = Vector{Float64}(undef, total_known_sites) # store imputed samples
    x_wmads = zeros(Float64, total_known_sites)#Vector{Float64}(undef, total_known_sites)
    original_mps_length = length(mps)

    last_impute_idx = 0 
    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            
            # pretty sure calling orthogonalize is a massive computational bottleneck
            #orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            # rdm = prime(A, s[i]) * dag(A)
            known_state_as_ITensor = timeseries_enc[i]
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc == total_known_sites
                A_new = mps[last_impute_idx] * Am # will IndexError if there are no sites to impute
            else
                A_new = mps[(site_loc+1)] * Am
            end
            # proba_state = get_conditional_probability(known_x, matrix(rdm), opts) # state' * rdm * state
            # A_new *= 1/sqrt(proba_state)
            normalize!(A_new)

            # if !isapprox(norm(A_new), 1.0)
            #     error("Site not normalised")
            # end
            
            mps[site_loc] = ITensor(1)
            if site_loc == total_known_sites
                mps[last_impute_idx] = A_new 
            else
                mps[site_loc + 1] = A_new
            end
        else
            last_impute_idx = i
        end
    end

    # collapse the mps to just the imputed sites
    mps_el = [tens for tens in mps if ndims(tens) > 0]
    mps = MPS(mps_el)
    s = siteinds(mps)

    inds = eachindex(mps)
    # inds = reverse(inds)
    orthogonalize!(mps, first(inds)) #TODO: this line is what breaks imputations of non adjacent sites, fix
    A = mps[first(inds)]
    for (ii,i) in enumerate(inds)
        rdm = prime(A, s[i]) * dag(A)
        # get previous ind
        if isassigned(x_samps, imputation_sites[i] - 1) # isassigned can handle out of bounds indices
            x_prev = x_samps[imputation_sites[i] - 1]

        elseif isassigned(x_samps, imputation_sites[i]+1)
            x_prev = x_samps[imputation_sites[i]+1]

        else
            x_prev = nothing
        end

        # mx, ms, mad = get_median_from_rdm(matrix(rdm), opts, enc_args; binary_thresh=dx, get_wmad=get_wmad) # dx = 0.001 by default
        mx, ms, mad = get_median_from_rdm(rdm, xvals, xvals_enc, s[i], opts, enc_args; get_wmad=get_wmad)
        x_samps[imputation_sites[i]] = mx
        x_wmads[imputation_sites[i]] = mad
       
        if ii != num_imputation_sites
            # sampled_state_as_ITensor = itensor(ms, s[i])
            ms = itensor(ms, s[i])
            #proba_state = get_conditional_probability(ms, rdm)
            Am = A * dag(ms)
            A_new = mps[inds[ii+1]] * Am
            normalize!(A_new)
            #A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end 
    return (x_samps, x_wmads)
end







function forward_impute_trajectory(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options,
        enc_args::AbstractVector; 
        check_rdm::Bool=false
    )

    """Applies inverse transform sampling to obtain samples from the conditional 
    rdm and returns a single realisation/shot resulting from sequentially sampling
    each site."""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the sampled x and sampled state from the rdm using inverse transform sampling
        sx, ss = get_sample_from_rdm(matrix(rdm), opts, enc_args)
        x_samps[i] = sx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ss, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(sx, matrix(rdm), opts, enc_args)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_impute_trajectory_time_dependent(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options, 
        enc_args::AbstractVector; 
        check_rdm::Bool=false
    )
    # same as above, but for time dep. encoding

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the sampled x and sampled state from the rdm using inverse transform sampling
        sx, ss = get_sample_from_rdm(matrix(rdm), opts, enc_args, i)
        x_samps[i] = sx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ss, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(sx, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_impute_directMean(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options,
        enc_args::AbstractVector; 
        check_rdm::Bool=false
    )
    """Evaluate different values of x for the conditional PDF and select the 
    mean (expectation) + variance"""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
        x_stds[i] = 0.0
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args)
        x_samps[i] = ex
        x_stds[i] = stdx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(es, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps, x_stds

end

function forward_impute_directMean_time_dependent(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options, 
        enc_args::AbstractVector;
        check_rdm::Bool=false
    )
    # time dep version

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
        x_stds[i] = 0.0
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args, i)
        x_samps[i] = ex
        x_stds[i] = stdx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(es, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps, x_stds

end

function forward_impute_directMode(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options,
        enc_args::AbstractVector; 
        check_rdm::Bool=false
    )
    """Evaluate different values of x for the conditional PDF and select the 
    mode."""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
   
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        # get the x value as the mode of the rdm, mx, and the corresponding state, ms, directly
        mx, ms = get_cpdf_mode(matrix(rdm), opts, enc_args)
        x_samps[i] = mx
       
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ms, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(mx, matrix(rdm), opts, enc_args)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_impute_directMode_time_dependent(
        class_mps::MPS, 
        known_values::Vector{Float64},
        forecast_start_site::Int, 
        opts::Options, 
        enc_args::AbstractVector;
        check_rdm::Bool=false
    )
    # time dependent version
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
   
    end
    # impute the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first impute. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        # get the x value as the mode of the rdm, mx, and the corresponding state, ms, directly
        mx, ms = get_cpdf_mode(matrix(rdm), opts, enc_args, i)
        x_samps[i] = mx
       
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ms, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(mx, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function any_impute_trajectory(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::Vector{Float64},
        imputation_sites::Vector{Int},
    )
    """impute mps sites without respecting time ordering, i.e., 
    condition on all known values first, then impute remaining sites one-by-one.
    Use inverse transform sampling to get a single trajectory."""
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            # pretty sure calling orthogonalize is the computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            # if the next site exists, absorb the previous tensor
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                # if at the end of the mps, absorb into previous site
                A_new = mps[(site_loc-1)] * Am
            end
            # normalise by the probability
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args)
            A_new *= 1/sqrt(proba_state)

            # check the norm 
            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            # make a new mps out of the remaining un-measured sites
            current_site = site_loc

            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end
    # sample from the remaining mps which contains unmeasured (imputation) sites. 
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end
    # place the mps into right canonical form
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        # same as for regular forecsting/sampling
        rdm = prime(A, s[i]) * dag(A)
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), opts, enc_args)
        x_samps[imputation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), opts, enc_args)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end

function any_impute_trajectory_time_dependent(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::Vector{Float64}, 
        imputation_sites::Vector{Int}
    )

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            # pretty sure calling orthogonalize is the computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args, i)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            # if the next site exists, absorb the previous tensor
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                # if at the end of the mps, absorb into previous site
                A_new = mps[(site_loc-1)] * Am
            end
            # normalise by the probability
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
            A_new *= 1/sqrt(proba_state)

            # check the norm 
            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            # make a new mps out of the remaining un-measured sites
            current_site = site_loc

            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end
    # sample from the remaining mps which contains unmeasured (imputation) sites. 
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end
    # place the mps into right canonical form
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        # same as for regular forecsting/sampling
        rdm = prime(A, s[i]) * dag(A)
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), opts, enc_args, i)
        x_samps[imputation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end

function any_impute_directMean(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::Vector{Float64},
        imputation_sites::Vector{Int}
    )
    """impute mps sites without respecting time ordering, i.e., 
    condition on all known values first, then impute remaining sites one-by-one.
    
    Use direct mean/variance"""
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            x_stds[i] = 0.0
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            current_site = site_loc
            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args)
        x_samps[imputation_sites[count]] = ex
        x_stds[imputation_sites[count]] = stdx
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(es, s[i])
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps, x_stds
end

function any_impute_directMean_time_dependent(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::Vector{Float64}, 
        imputation_sites::Vector{Int}
        )
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            x_stds[i] = 0.0
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args, i)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            current_site = site_loc
            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args, i)
        x_samps[imputation_sites[count]] = ex
        x_stds[imputation_sites[count]] = stdx
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(es, s[i])
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps, x_stds
end



function any_impute_directMode(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::AbstractVector{<:Number}, 
        timeseries_enc::MPS,
        imputation_sites::Vector{Int}; 
        mode_range::Tuple{<:Number, <:Number}=opts.encoding.range, 
        dx::Float64=1E-4, 
        xvals::Vector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        max_jump::Union{Number,Nothing}=0.5
    )

    """impute mps sites without respecting time ordering, i.e., 
    condition on all known values first, then impute remaining sites one-by-one.
    Use direct mode."""
    if isempty(imputation_sites)
        throw(ArgumentError("imputation_sites can't be empty!")) 
    end
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    total_known_sites = length(mps)
    num_imputation_sites = length(imputation_sites)
    x_samps = Vector{Float64}(undef, total_known_sites)
    original_mps_length = length(mps)

    last_impute_idx = 0 
    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            
            # pretty sure calling orthogonalize is a massive computational bottleneck
            #orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            # rdm = prime(A, s[i]) * dag(A)
            known_state_as_ITensor = timeseries_enc[i]
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc == total_known_sites
                A_new = mps[last_impute_idx] * Am # will IndexError if there are no sites to impute
            else
                A_new = mps[(site_loc+1)] * Am
            end
            # proba_state = get_conditional_probability(known_x, matrix(rdm), opts) # state' * rdm * state
            # A_new *= 1/sqrt(proba_state)
            normalize!(A_new)

            # if !isapprox(norm(A_new), 1.0)
            #     error("Site not normalised")
            # end
            
            mps[site_loc] = ITensor(1)
            if site_loc == total_known_sites
                mps[last_impute_idx] = A_new 
            else
                mps[site_loc + 1] = A_new
            end
        else
            last_impute_idx = i
        end
    end

    # collapse the mps to just the imputed sites
    mps_el = Vector{ITensor}(undef, num_imputation_sites)
    i = 1
    for tens in mps
        if ndims(tens) > 0
            mps_el[i] = tens # WHYY is MPS not broadcastable ?!??!!
            i += 1
        end
    end
    mps = MPS(mps_el)
    s = siteinds(mps)

    inds = eachindex(mps)
    # inds = reverse(inds)
    orthogonalize!(mps, first(inds)) #TODO: this line is what breaks imputations of non adjacent sites, fix
    A = mps[first(inds)]
    for (ii,i) in enumerate(inds)
        rdm = prime(A, s[i]) * dag(A)
        # get previous ind
        if isassigned(x_samps, imputation_sites[i] - 1) # isassigned can handle out of bounds indices
            x_prev = x_samps[imputation_sites[i] - 1]

        elseif isassigned(x_samps, imputation_sites[i]+1)
            x_prev = x_samps[imputation_sites[i]+1]

        else
            x_prev = nothing
        end

        mx, ms = get_cpdf_mode(rdm, xvals, xvals_enc, s[i], opts, enc_args, x_prev, max_jump)

        x_samps[imputation_sites[i]] = mx
       
        if ii != num_imputation_sites
            # sampled_state_as_ITensor = itensor(ms, s[i])
            proba_state = get_conditional_probability(ms, rdm)
            Am = A * dag(ms)
            A_new = mps[inds[ii+1]] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end 
    return x_samps
end

function any_impute_directMode_time_dependent(
        class_mps::MPS, 
        opts::Options, 
        enc_args::AbstractVector,
        timeseries::Vector{Float64}, 
        imputation_sites::Vector{Int}
    )

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), imputation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = timeseries[i]
            x_samps[i] = known_x
            
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args, i)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            current_site = site_loc
            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        mx, ms = get_cpdf_mode(matrix(rdm), opts, enc_args, i)
        x_samps[imputation_sites[count]] = mx
       
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(ms, s[i])
            proba_state = get_conditional_probability(mx, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps
end

"""
Impute a SINGLE trajectory using inverse transform sampling (ITS).\n
"""
function any_impute_ITS_single(
    class_mps::MPS, 
    opts::Options, 
    enc_args::AbstractVector,
    timeseries::AbstractVector{<:Number},
    timeseries_enc::MPS, 
    imputation_sites::Vector{Int};
    atol=1e-5,
    rejection_threshold::Union{Float64, Symbol}=1.0)

    if isempty(imputation_sites)
        throw(ArgumentError("Imputation sites cannot be empty!"))
    end

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    total_known_sites = length(mps)
    known_sites = setdiff(collect(1:total_known_sites), imputation_sites)
    num_imputation_sites = length(imputation_sites)
    x_samps = similar(timeseries, Float64)
    original_mps_length = length(mps)
    
    last_impute_idx = 0
    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            site_loc = findsite(mps, s[i])
            known_x = timeseries[i]
            x_samps[i] = known_x
            
            A = mps[site_loc]
            known_state_as_ITensor = timeseries_enc[i]
            # make projective measurement by contracting state vector with the MPS site
            Am = A * dag(known_state_as_ITensor)
            if site_loc == total_known_sites
                # check if at the last site
                A_new = mps[last_impute_idx] * Am # will IndexError if there are no sites to impute
            else
                # absorb into next site
                A_new = mps[(site_loc+1)] * Am
            end
            normalize!(A_new)

            mps[site_loc] = ITensor(1)
            if site_loc == total_known_sites
                mps[last_impute_idx] = A_new
            else
                mps[site_loc + 1] = A_new
            end
        else
            last_impute_idx = i
        end
    end

    # collapse the mps to just the imputed sites
    mps_el = [tens for tens in mps if ndims(tens) > 0]
    mps = MPS(mps_el)
    s = siteinds(mps)

    inds = eachindex(mps)
    orthogonalize!(mps, first(inds))
    A = mps[first(inds)]
    for (ii,i) in enumerate(inds)
        rdm = prime(A, s[i]) * dag(A)
        # get prev ind
        if isassigned(x_samps, imputation_sites[i] - 1)
            x_prev = x_samps[imputation_sites[i] - 1]
        elseif isassigned(x_samps, imputation_sites[i]+1)
            x_prev = x_samps[imputation_sites[i]+1]
        else
            x_prev = nothing
        end
        # sample a value/state from the conditional probability distribution using ITS
        sx, ss = get_sample_from_rdm(matrix(rdm), opts, enc_args; atol=atol, threshold=rejection_threshold)
        
        x_samps[imputation_sites[i]] = sx
        if ii != num_imputation_sites
            ss = itensor(ss, s[i])
            proba_state = get_conditional_probability(ss, rdm)
            Am = A * dag(ss)
            A_new = mps[inds[ii+1]] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end
    return x_samps 
end
