
mutable struct EncodedDataRange
    dx::Float64
    guess_range::Tuple{R,R} where R <: Real
    xvals::Vector{Float64}
    site_index::Index
    xvals_enc::Vector{<:AbstractVector{<:AbstractVector{<:Number}}} # https://i.imgur.com/cmFIJmS.png (I apologise)
end 

mutable struct ImputationProblem
    mpss::AbstractVector{<:MPS}
    X_train::Matrix{<:Real}
    y_train::Vector{<:Integer}
    X_test::Matrix{<:Real}
    y_test::Vector{<:Integer}
    opts::Options
    enc_args::Vector{Any}
    x_guess_range::EncodedDataRange
    class_map::Dict{Any, Int}
end

# probably redundant if enc args are provided externally from training
function get_enc_args_from_opts(
        opts::Options, 
        X_train::Matrix, 
        y::Vector{Int};
        verbosity::Integer=1.
    )
    """Rescale and then Re-encode the scaled training data using the time dependent
    encoding to get the encoding args."""

    X_train_scaled, norm = transform_train_data(permutedims(X_train) ;opts=opts)

    
    if isnothing(opts.encoding.init)
        enc_args = []
    else
        verbosity >= 1 && println("Re-encoding the training data to get the encoding arguments...")
        order = sortperm(y)
        enc_args = opts.encoding.init(X_train_scaled[:,order], y[order]; opts=opts)
    end

    return enc_args

end


function init_imputation_problem(
        mps::MPS, 
        X_train::AbstractMatrix{R}, 
        y_train::AbstractVector{Int}, 
        X_test::AbstractMatrix{R}, 
        y_test::AbstractVector{Int},
        opts::AbstractMPSOptions; 
        verbosity::Integer=1,
        dx::Float64 = 1e-4,
        guess_range::Union{Nothing, Tuple{R,R}}=nothing,
        static_xvecs::Bool=true
    ) where {R <: Real}
    """No saved JLD File, just pass in variables that would have been loaded 
    from the jld2 file. Need to pass in reconstructed opts struct until the 
    issue is resolved."""

    if opts isa MPSOptions
        opts = Options(opts)
    end

    if isnothing(guess_range)
        guess_range = opts.encoding.range
    end


    # extract info
    verbosity > 0 && println("+"^60 * "\n"* " "^25 * "Summary:\n")
    verbosity > 0 && println(" - Dataset has $(size(X_train, 1)) training samples and $(size(X_test, 1)) testing samples.")
    verbosity > 0 && println("Slicing MPS into individual states...")
    mpss, label_idx = expand_label_index(mps)
    num_classes = length(mpss)
    verbosity > 0 && println(" - $num_classes class(es) were detected.")

    if opts.encoding.istimedependent
        verbosity > 0 && println(" - Time dependent encoding - $(opts.encoding.name) - detected")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        verbosity > 0 && println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train; verbosity=verbosity)

    xvals=collect(range(guess_range...; step=dx))
    site_index=Index(opts.d)
    if opts.encoding.istimedependent
        verbosity > -2 && println("Pre-computing possible encoded values of x_t, this may take a while... ")
        # be careful with this variable, for d=20, length(mps)=100, this is nearly 1GB for a basis that returns complex floats
        if static_xvecs
            xvals_enc = [[SVector{opts.d}(get_state(x, opts, j, enc_args)) for x in xvals] for j in 1:length(mps)] # a proper nightmare of preallocation, but necessary
        else
            xvals_enc = [[(x, opts, j, enc_args) for x in xvals] for j in 1:length(mps)] # a proper nightmare of preallocation, but necessary
        end
    else
        if static_xvecs
            xvals_enc_single = [SVector{opts.d}(get_state(x, opts, 1, enc_args)) for x in xvals]
        else
            xvals_enc_single = [get_state(x, opts, 1, enc_args) for x in xvals]
        end
        xvals_enc = [view(xvals_enc_single, :) for _ in 1:length(mps)]
    end

    x_guess_range = EncodedDataRange(dx, guess_range, xvals, site_index, xvals_enc)
    mpss, l_ind = expand_label_index(mps)

    classes_unique = sort(unique(y_train))
    class_map = Dict{Int, Any}()
    for (i, class) in enumerate(classes_unique)
        class_map[class] = i
    end
    imp_prob = ImputationProblem(mpss, X_train, y_train, X_test, y_test, opts, enc_args, x_guess_range, class_map);

    verbosity > 0 && println("\n Created $num_classes ImputationProblem struct(s) containing class-wise mps and test samples.")

    return imp_prob

end

"""
    init_imputation_problem(W::TrainedMPS, X_test::AbstractMatrix, y_test::AbstractArray=zeros(Int, size(X_test,1)), [custom_encoding::MPSTime.Encoding]; <keyword arguments>) -> imp::ImputationProblem
    init_imputation_problem(W::TrainedMPS, X_test::AbstractMatrix, [custom_encoding::MPSTime.Encoding]); <keyword arguments>) -> imp::ImputationProblem


Initialise an imputation problem using a trained MPS and relevent test data.

This involves a lot of pre-computation, which can be quite time intensive for data-driven bases. For unclassed/unsupervised data `y_test` may be omitted. 
If the MPS was trained with a custom encoding, then this encoding must be passed to `init_imputation_problem`.

# Keyword Arguments
- `guess_range::Union{Nothing, Tuple{<:Real,<:Real}}=nothing`: The range of values that guesses are allowed to take. This range is applied to normalised, encoding-adjusted time-series data. To allow any guess, leave as nothing, or set to encoding.range (e.g. [(-1., 1.) for the legendre encoding]).
- `dx::Float64 = 1e-4`: The spacing between possible guesses in normalised, encoding-adjusted units. When imputing missing data with an MPS method, the imputed values will be selected from 
    range(guess_range...; step=dx)
- `verbosity::Integer=1`: The verbosity of the initialisation process. Useful for debugging, or to completely suppress output.
- `test_encoding::Bool=true`: Whether to double check the encoding and scaling options are correct. This is strongly recommended but has a slight performance cost, so may be disabled.
- `static_xvecs::Bool=true`: Whether to store encoded xvalues as StaticVectors. Usually improved performance
"""
function init_imputation_problem(
    mps::TrainedMPS, 
    X_test::AbstractMatrix, 
    y_test::AbstractVector=zeros(Int, size(X_test,1)), 
    custom_encoding::Union{Encoding, Nothing}=nothing; 
    test_encoding::Bool=true, 
    kwargs...)

    X_train = mps.train_data.original_data
    y_train = [ts.label for ts in mps.train_data.timeseries]
    opts = mps.opts
    opts_concrete = safe_options(opts) # make sure options isnt abstract

    if !isnothing(custom_encoding)
        if !(opts.encoding in [:custom, :Custom])
            throw(ArgumentError("To impute with a custom encoding, the MPS must have been trained with a custom encoding. See the details of the \'encoding = :Custom\' setting in MPSOptions"))
        else
            opts_concrete = _set_options(opts_concrete; encoding=custom_encoding)
        end
    end

    # test that nothing has gone wrong with the encoding
    if test_encoding
        X_train_scaled, norms = transform_train_data(permutedims(X_train); opts=opts_concrete)

        classes = unique(vcat(y_train))
        num_classes = length(classes)

        sort!(classes)
        class_keys = Dict(zip(classes, 1:num_classes)) # why did I write encode_dataset in this way? (#TODO move the definition of of class_keys inside encode_dataset)
                                                       # TODO TODO use scientific types to avoid the problem entirely
    
        
        s = EncodeSeparate{opts.encode_classes_separately}()
        sites = get_siteinds(mps.mps)
        training_states, enc_args_tr = encode_dataset(s, X_train, X_train_scaled, y_train, "train", sites; opts=opts_concrete, class_keys=class_keys)     
        if !isapprox(training_states, mps.train_data)

            if isnothing(custom_encoding)
                error("Could not reproduce the encoded training set from the TrainedMPS. This should never happen, has there been some data corruption?")
            else
                error("Could not reproduce the encoded training set from the TrainedMPS, double check that custom_encoding matches the encoding the MPS was trained with. Otherwise, this should never happen, has there been some data corruption?")
            end
        end
    end

    return init_imputation_problem(mps.mps, X_train, y_train, X_test, y_test, opts_concrete; kwargs...)
end


function init_imputation_problem(mps::TrainedMPS, X_test::AbstractMatrix, custom_encoding::Encoding; kwargs...)

    return init_imputation_problem(mps, X_test, zeros(Int, size(X_test,1)), custom_encoding; kwargs...)
end



"""
```Julia
kNN_impute(imp::ImputationProblem, 
           [class::Any,] 
           instance::Integer, 
           missing_sites::AbstractVector{<:Integer}; 
           k::Integer=1) -> [neighbour1::Vector, neighbour2::Vector, ...]
```
Impute `missing_sites` using the `k` nearest neighbours in the test set, based on Euclidean distance.

See [`init_imputation_problem`](@ref) for constructing an ImputationProblem instance. The `instance` number is relative to the class, so
class 1, instance 2 would be distinct from class 2, instance 2.
"""
function kNN_impute(
        imp::ImputationProblem,
        class::Any, 
        instance::Integer, 
        missing_sites::AbstractVector{<:Integer}; 
        k::Integer=1
    )

    mps = imp.mpss[imp.class_map[class]]
    X_train = imp.X_train
    y_train = imp.y_train

    cl_inds = (1:length(imp.y_test))[imp.y_test .== class] # For backwards compatibility reasons
    target_timeseries_full = imp.X_test[cl_inds[instance], :]

    known_sites = setdiff(collect(1:length(mps)), missing_sites)
    target_series = target_timeseries_full[known_sites]

    c_inds = findall(y_train .== class)
    Xs_comparison = X_train[c_inds, known_sites]

    mses = Vector{Float64}(undef, length(c_inds))

    for (i, ts) in enumerate(eachrow(Xs_comparison))
        mses[i] = (ts .- target_series).^2 |> mean
    end
    
    min_inds = partialsortperm(mses, 1:k)
    ts = Vector(undef, k)

    for (i,min_ind) in enumerate(min_inds)
        ts_ind = c_inds[min_ind]
        ts[i] = X_train[ts_ind,:]
    end


    return ts


end
function kNN_impute(
        imp::ImputationProblem,
        instance::Integer, 
        missing_sites::AbstractVector{<:Integer}; 
        k::Integer=1
    )
    return kNN_impute(imp, 0, instance, missing_sites;k=k)
end

function get_predictions(
        imp::ImputationProblem,
        class::Any, 
        instance::Integer, 
        missing_sites::Vector{<:Integer}, 
        method::Symbol=:median;
        impute_order::Symbol=:forwards,
        invert_transform::Bool=true, # whether to undo the sigmoid transform/minmax normalisation, if this is false, timeseries that hve extrema larger than any training instance may give odd results
        kwargs... # method specific keyword arguments
    )

    # setup imputation variables
    X_test = imp.X_test
    X_train = imp.X_train

    mps = imp.mpss[imp.class_map[class]]
    cl_inds = (1:length(imp.y_test))[imp.y_test .== class] # For backwards compatibility reasons
    target_ts_raw = X_test[cl_inds[instance], :]
    target_timeseries= deepcopy(target_ts_raw)

    # transform the data
    # perform the scaling

    X_train_scaled, norms = transform_train_data(X_train; opts=imp.opts)
    target_timeseries_full, oob_rescales_full = transform_test_data(target_ts_raw, norms; opts=imp.opts)

    target_timeseries[missing_sites] .= mean(X_train[:]) # make it impossible for the unknown region to be used, even accidentally
    target_timeseries, oob_rescales = transform_test_data(target_timeseries, norms; opts=imp.opts)

    sites = get_siteinds(mps)
    target_enc = MPS([itensor(get_state(x, imp.opts, j, imp.enc_args), sites[j]) for (j,x) in enumerate(target_timeseries)])

    pred_err = []
    if method == :mean       
        ts, pred_err = impute_mean(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, missing_sites; impute_order=impute_order, kwargs...)
        ts = [ts] # type stability
        pred_err = [pred_err]

    elseif method == :median
        ts, pred_err = impute_median(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, missing_sites; impute_order=impute_order, kwargs...)
        ts = [ts] # type stability
        pred_err = [pred_err]

    elseif method == :mode
        ts = impute_mode(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, missing_sites; impute_order=impute_order, kwargs...)
        ts = [ts] # type stability

    elseif method == :ITS
        ts = impute_ITS(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, missing_sites; impute_order=impute_order, kwargs...)

    elseif method ==:kNearestNeighbour
        ts = kNN_impute(imp, class, instance, missing_sites; kwargs...)

        if !invert_transform
            for i in eachindex(ts)
                ts[i], _ = transform_test_data(ts[i], norms; opts=imp.opts)
            end
        end

    else
        error("Invalid method. Choose :mean, :mode, :median, :kNearestNeighbour, or :ITS")
    end


    if invert_transform && !(method == :kNearestNeighbour)
        if !isempty(pred_err)
            for i in eachindex(ts)
                pred_err[i] .+=  ts[i] # add the time-series, so nonlinear rescaling is reversed correctly
                ts[i] = invert_test_transform(ts[i], oob_rescales, norms; opts=imp.opts)


                try
                    pred_err[i] = invert_test_transform(pred_err[i], oob_rescales, norms; opts=imp.opts)
                catch e
                    if e isa DomainError
                        @warn "Imputation error was too large and could not be transformed back into unnormalised units, returning problematic error values as NaNs. Try tuning your hyperparameters. The uncorrected error can be viewed in normalised space by passing invert_transform=false to MPS_impute."
                        max_retries = length(pred_err[i])
                        problematic_errors = []
                        j = 1
                        success = false
                        while j <= max_retries # this needs to generalise to any basis / any kind of normalisation, so we have to remove values and try to get invert_test_transform to work
                            ei = argmax(abs.(pred_err[i])) 
                            push!(problematic_errors, ei)
                            pred_err[i][ei] = mean(ts[i]) # the scale is unknown, so this is the safest value I can think to use 
                            safe = true
                            try
                                pred_err[i] = invert_test_transform(pred_err[i], oob_rescales, norms; opts=imp.opts)
                            catch e2
                                if e2 isa DomainError
                                    safe = false
                                else
                                    throw(e)
                                end
                            end

                            if safe
                                success = true
                                break
                            end
                
                            j += 1
                        end

                        if !success
                            @warn "All of the errors were too large!"
                        end

                        pred_err[i][problematic_errors] .= NaN
                    else
                        throw(e)
                    end
                end

                pred_err[i] .-=  ts[i] # remove the time-series, leaving the unscaled uncertainty          
            end
            
        else
            for i in eachindex(ts)
                ts[i] = invert_test_transform(ts[i], oob_rescales, norms; opts=imp.opts)            
            end

        end

        target = target_ts_raw

    elseif method == :kNearestNeighbour
        target = target_ts_raw
    else
        target = target_timeseries_full
    end

    if isempty(pred_err)
        pred_err = [nothing for _ in eachindex(ts)] # helps the plotting functions not crash later
    end
    

    return ts, pred_err, target
end



"""
```Julia
MPS_impute(imp::ImputationProblem, 
           [class::Any,] 
           instance::Integer, 
           missing_sites::AbstractVector{<:Integer}, 
           method::Symbol=:median; 
           <keyword arguments>) -> (imputed_instances::Vector, errors::Vector, targets::Vector, stats::Dict, plots::Vector{Plots.Plot})
```
Impute the `missing_sites` using an MPS-based approach, selecting the trajectory from the conditioned distribution with `method`. 

The imputed time series, the imputation errors, the original time series, statistics about goodness of fit, and any plots generated by `MPS_impute`\
 are returned by `imputed_instances`, `errors`, `targets`, `stats`, and `plots` respectively. These will always be length one vectors,\
 unless the `ITS` or `kNearestNeighbour` method are used, which may cause `MPS_impute` to return multiple potential imputations with the associated errors, stats, and plots.

See [`init_imputation_problem`](@ref) for constructing an `ImputationProblem` instance out of a trained MPS. The `instance` number is relative to the class, so
class 1, instance 2 would be distinct from class 2, instance 2.

# Imputation Methods
- `:median`: For each missing value, compute the probability density function of the possible outcomes from the MPS, and choose the median. This method is the most robust to outliers. Keywords:
    * `get_wmad::Bool=true`: Whether to return an 'error' vector that computes the Weighted Median Absolute Deviation (WMAD) of each imputed value.

- `:mean`: For each missing value, compute the probability density function of the possible outcomes from the MPS, and choose the expected value. Keywords:
    * `get_std::Bool=true`: Whether to return an 'error' vector that computes standard deviation of each imputed value.

- `:mode`: For each missing value, choose the most likely outcome predicted by the MPS. Keywords:
    * `max_jump::Union{Number,Nothing}=nothing`: The largest jump allowed between two adjacent imputations. Leave as `nothing` to allow any jump. Helpful to suppress 'spikes' caused by poor support near the encoding domain edges.

- `:ITS`: For each missing value, choose a value at random with probability weighted by the probability density function of the possible outcomes. Keywords:
    * `rseed::Integer=1`: Random seed for producing the trajectories.
    * `num_trajectories::Integer=1`: Number of trajectories to compute.
    * `rejection_threshold::Union{Float64, Symbol}=:none`: Number of WMADs allowed between adjacent points. Setting this low helps suppress rapidly varying trajectories that occur by bad luck. 
    * `max_trials::Integer=10`: Number of attempts allowed to make guesses conform to rejection_threshold before giving up.

- `:kNearestNeighbour`: Select the `k` nearest neighbours in the training set using Euclidean distance to the known data. Keyword:
    * `k`: Number of nearest neighbours to return. See [`kNN_impute`](@ref)

# Keyword Arguments
- `impute_order::Symbol=:forwards`: Whether to impute the missing values `:forwards` (left to right) or `:backwards` (right to left)
- `NN_baseline::Bool=true`: Whether to also impute the missing data with a k-Nearest Neighbour baseline.
- `n_baselines::Integer=1`: How many nearest neighbour baselines to compute.
- `plot_fits::Bool=true`: Whether to make a plot showing the target timeseries, the missing values, and the imputed region. If false, then p will be an empty vector. The plot will show the NN_baseline (if it was computed), as well as every trajectory if using the :ITS method.
- `get_metrics::Bool=true`: Whether to compute imputation metrics, if false, then `stats`, will be empty.
- `full_metrics::Bool=false`: Whether to compute every metric (MAPE, SMAPE, MAE, MSE, RMSE) or just MAE and MAPE.
- `print_metric_table::Bool=false`: Whether to print the `stats` as a table.
- `invert_transform::Bool=true`:, # Whether to undo the sigmoid transform/minmax normalisation before returning the imputed points. If this is false, imputed_instance, errors, target timeseries, stats, and plot y-axis will all be scaled by the data preprocessing / normalisation and fit to the encoding domain.
- `kwargs...`: Extra keywords passed to `MPS_impute` are forwarded to the imputation method. See the Imputation Methods section.
"""
function MPS_impute(
        imp::ImputationProblem,
        class::Any, 
        instance::Integer, 
        missing_sites::Vector{Int}, 
        method::Symbol=:median;
        invert_transform::Bool=true,
        impute_order::Symbol=:forwards,
        NN_baseline::Bool=true, 
        n_baselines::Integer=1,
        plot_fits=true,
        get_metrics::Bool=true, # whether to compute goodness of fit metrics
        full_metrics::Bool=false, # whether to compute every metric or just MAE
        print_metric_table::Bool=false,
        kwargs... # passed on to the imputer that does the real work
    )


    mps = imp.mpss[imp.class_map[class]]
    chi_mps = maxlinkdim(mps)
    d_mps = get_siteinds(mps)[1] |> ITensors.dim
    enc_name = imp.opts.encoding.name

    ts, pred_err, target = get_predictions(imp, class, instance, missing_sites, method; invert_transform=invert_transform, impute_order=impute_order, kwargs...)

    if plot_fits
        p1 = plot(ts[1], ribbon=pred_err[1], xlabel="time", ylabel="x", 
            label="MPS imputed", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm
        )

        for i in eachindex(ts)[2:end]
            p1 = plot!(ts[i], ribbon=pred_err[i], label="MPS imputed $i", ls=:dot, lw=2, alpha=0.8)
        end

        p1 = plot!(target, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
        p1 = title!("Sample $instance, Class $class, $(length(missing_sites))-site Imputation, 
            d = $d_mps, χ = $chi_mps, $enc_name encoding"
        )
        ps = [p1] # for type stability
    else
        ps = []
    end


    metrics = []
    if get_metrics
        for t in ts
            if full_metrics
                push!(metrics, compute_all_forecast_metrics(t[missing_sites], target[missing_sites], print_metric_table))
            else
                push!(metrics, Dict(:MAE => mae(t[missing_sites], target[missing_sites]), 
                    :MAPE => mape(t[missing_sites], target[missing_sites])))
            end
        end
    end

    if NN_baseline
        mse_ts, _... = get_predictions(imp, class, instance, missing_sites, :kNearestNeighbour; invert_transform=invert_transform, k=n_baselines)

        if plot_fits 
            for (i,t) in enumerate(mse_ts)
                p1 = plot!(t, label="Nearest Train Data $i", c=:red,lw=2, alpha=0.7, ls=:dot)
            end

            ps = [p1] # for type stability
        end

        
        if get_metrics
            if full_metrics # only compute the first NN_MAE
                NN_metrics = compute_all_forecast_metrics(mse_ts[1][missing_sites], target[missing_sites], print_metric_table)
                for key in keys(NN_metrics)
                    metrics[1][Symbol("NN_" * string(key) )] = NN_metrics[key]
                end
            else
                metrics[1][:NN_MAE] = mae(mse_ts[1][missing_sites], target[missing_sites])
                metrics[1][:NN_MAPE] = mape(mse_ts[1][missing_sites], target[missing_sites])
            end
        end
    end

    return ts, pred_err, target, metrics, ps
end

# classless method
function MPS_impute(
        imp::ImputationProblem,
        instance::Integer, 
        missing_sites::Vector{Int}, 
        method::Symbol=:median;
        kwargs... 
    )

    return MPS_impute(imp, 0, instance, missing_sites, method; kwargs...)

end


"""
```Julia
get_cdfs(imp::ImputationProblem, 
           class::Any, 
           instance::Integer, 
           missing_sites::AbstractVector{<:Integer}, 
           method::Symbol=:median; 
           <keyword arguments>) -> (cdfs::Vector{Vector}, ts::Vector, pred_err::Vector, target_timeseries_full::Vector)
```
Impute the `missing_sites` using an MPS-based approach, selecting the trajectory from the conditioned distribution with `method`, and returns the cumulative distribution function used to infer each missing value. 

See [`MPS_impute`](@ref) for a list of imputation methods and keyword arguments (does not support plotting, stats, or kNN baselines). See [`init_imputation_problem`](@ref) for constructing an ImputationProblem instance. The `instance` number is relative to the class, so
class 1, instance 2 would be distinct from class 2, instance 2.

"""
function get_cdfs(
        imp::ImputationProblem,
        class::Any, 
        instance::Integer, 
        missing_sites::Vector{Int};
        kwargs... # method specific keyword arguments
    )

    # setup imputation variables
    X_test = imp.X_test
    X_train = imp.X_train

    mps = imp.mpss[imp.class_map[class]]
    cl_inds = (1:length(imp.y_test))[imp.y_test .== class] # For backwards compatibility reasons
    target_ts_raw = imp.X_test[cl_inds[instance], :]
    target_timeseries = deepcopy(target_ts_raw)

    # transform the data
    # perform the scaling

    X_train_scaled, norms = transform_train_data(X_train; opts=imp.opts)
    target_timeseries_full, oob_rescales_full = transform_test_data(target_ts_raw, norms; opts=imp.opts)

    target_timeseries[missing_sites] .= mean(X_test[:]) # make it impossible for the unknown region to be used, even accidentally
    target_timeseries, oob_rescales = transform_test_data(target_timeseries, norms; opts=imp.opts)

    sites = get_siteinds(mps)
    target_enc = MPS([itensor(get_state(x, imp.opts, j, imp.enc_args), sites[j]) for (j,x) in enumerate(target_timeseries)])


    ts, pred_err, cdfs = get_rdms_with_med(mps, imp.opts, imp.x_guess_range, imp.enc_args, target_timeseries, target_enc, missing_sites; kwargs...)
    ts = [ts] # type stability
    pred_err = [pred_err]
    

    return cdfs, ts, pred_err, target_timeseries_full
end
