# gridsearch hyperparameter opt
using Base.Threads

include("../RealRealHighDimension.jl")
include("hyperUtils.jl")

using MLJBase: train_test_pairs, StratifiedCV

function hyperopt(encoding::Encoding, Xs::AbstractMatrix, ys::AbstractVector; 
    method="GridSearch", 
    etas::AbstractVector{<:Number}, 
    max_sweeps::Integer, 
    ds::AbstractVector{<:Integer}, 
    chi_maxs::AbstractVector{<:Integer}, 
    chi_init::Integer=4,
    train_ratio=0.9,
    force_complete_crossval::Bool=true, # overrides train_ratio
    nfolds::Integer= force_complete_crossval ? round(Int, ceil(1 / (1-train_ratio); digits=5)) : 1, # you can use this to override the number of folds you _should_ use, but don't. You need the round for floating point reasons
    mps_seed::Real=4567,
    kfoldseed::Real=1234567890, # overridden by the rng parameter
    foldrng::AbstractRNG=MersenneTwister(kfoldseed),
    update_iters::Integer=1,
    verbosity::Real=-1,
    dtype::Type = encoding.iscomplex ? ComplexF64 : Float64,
    loss_grad::Function=loss_grad_KLD,
    bbopt::BBOpt=BBOpt("CustomGD", "TSGO"),
    track_cost::Bool=false,
    rescale::Tuple{Bool,Bool}=(false, true),
    aux_basis_dim::Integer=2,
    encode_classes_separately::Bool=false,
    train_classes_separately::Bool=false,
    minmax::Bool=true,
    cutoff::Number=1e-10,
    force_overwrite::Bool=false,
    always_abort::Bool=false,
    dir::String="LogLoss/hyperopt/",
    distribute::Bool=true, # whether to destroy my ram or not
    exit_early::Bool=true
    )

    if force_overwrite && always_abort 
        error("You can't force_overwrite and always_abort that doesn't make any sense")
    end

    ########## Sanity checks ################
    if encoding.iscomplex
        if dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end


    @assert issorted(ds) "Hyperparamater vector \"ds\" is not sorted"
    @assert issorted(etas) "Hyperparamater vector \"etas\" is not sorted"
    @assert issorted(chi_maxs) "Hyperparamater vector \"chi_maxs\" is not sorted"


    # data is _input_ in python canonical (row major) format
    @assert size(Xs, 1) == size(ys, 1) "Size of training dataset and number of training labels are different!"

    


    ############### Data structures aand definitions ########################
    println("Allocating initial Arrays and checking for existing files")
    masteropts = Options(; nsweeps=max_sweeps, chi_max=1, d=1, eta=1, cutoff=cutoff, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
        bbopt=bbopt, track_cost=track_cost, rescale = rescale, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
        train_classes_separately=train_classes_separately, minmax=minmax, exit_early=exit_early)

    
    # Output files
    function repr_vec(v::AbstractVector) 
        c = minimum(abs.(diff(v)), init=1.)
        midstr = c == 1. ? "" : "$(c):"
        "$(first(v)):$(midstr)$(last(v))"
    end
    
    vstring = train_classes_separately ? "Split_train_" : ""
    pstr = encoding.name * "_" * vstring * "$(nfolds)fold_r$(mps_seed)_eta$(repr_vec(etas))_ns$(max_sweeps)_chis$(repr_vec(chi_maxs))_ds$(repr_vec(ds))"


    path = dir* pstr *"/"
    svfol = path*"data/"
    logpath = path # could change to a new folder if we wanted

    logfile = logpath * "log.txt"
    resfile = logpath  * "results.jld2"
    finfile = logpath * "finished.jld2"
    encodings = [encoding] # backwards compatibility reasons (/future feature)

    ######## Check if hyperopt already exists?  ################
    # resume if possible

if isdir(path) && !isempty(readdir(path))
    files = sort(readdir(path))
    safe_dir = all(files == sort(["log.txt", "results.jld2"])) || all(files == sort(["log.txt", "results.jld2", "finished.jld2"]))

    if !safe_dir
        error("Unknown (or missing) files in \"$path\". Move your data or it could get deleted!")
    end

    if isfile(finfile)
        if always_abort
            error("Aborting to conserve existing data")
        elseif !force_overwrite

            while true
                print("A hyperopt with these parameters already exists, overwrite the contents of \"$path\"? [y/n]: ")
                input = lowercase(readline())
                if input == "y"
                    break
                elseif input == "n"
                    error("Aborting to conserve existing data")
                end
            end
        end
        # Remove the saved files
        # the length is for safety so we can never recursively remove something terrible like "/" (shout out to the steam linux runtime)
        if isdir(path) && length(path) >=3 
            rm(logfile)
            rm(resfile)
            rm(finfile)
            rm(path; recursive=false) # safe because it will only remove empty directories 
        end
        results = Array{Union{Result,Missing}}(missing, nfolds,  max_sweeps+1, length(etas), length(ds), length(chi_maxs), length(encodings)) # Somewhere to save the results for no sweeps up to max_sweeps

    elseif isfile(resfile)
        resume = check_status(resfile, nfolds, etas, chi_maxs, ds, encodings)
        if resume
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
            done = Int(sum((!ismissing).(results)) / (max_sweeps+1))
            todo = Int(prod(size(results)) / (max_sweeps+1))
            println("Found interrupted benchmark with $(done)/$(todo) trains complete, resuming")

        else
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
            error("??? A status file exists but the parameters don't match!\nnfolds=$(nfolds_r)\netas=$(etas_r)\nns=$(max_sweeps_r)\nchis=$(chi_maxs_r)\nds=$(ds_r)")
        end
    else
        error("A non benchmark folder with the name\n$path\nAlready exists")
    end
else
    results = Array{Union{Result,Missing}}(missing, nfolds,  max_sweeps+1, length(etas), length(ds), length(chi_maxs), length(encodings)) # Somewhere to save the results for no sweeps up to max_sweeps
end

# make the folders and output file if they dont already exist
if !isdir(path) 
    mkdir(path)
    save_results(resfile, results, -1, nfolds, max_sweeps, -1., etas, -1, chi_maxs, -1, ds, first(encodings), encodings) 

    f = open(logfile, "w")
    close(f)
end



################### Definitions continued ##########################


    # all data concatenated for folding purposes
    # ntrs = round(Int, length(ys) *train_ratio)
    # nvals = length(ys) - ntrs

    num_mps_sites = size(Xs, 2)
    classes = unique(ys)
    num_classes = length(classes)


    # A few more checks
    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes)) # Assign each class a 'Key' from 1 to n
    

    Xs_train_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Training data for each dimension and each fold
    Xs_val_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Validation data for each dimension and each fold


    sites = Array{Vector{Index{Int64}}}(undef, num_mps_sites, length(ds)) # Array to hold the siteindices for each dimension
    Ws = Vector{MPS}(undef, length(ds)) # stores an MPS for each encoding dimension



    ############### generate the starting MPS for each d ####################
    println("Generating $(length(ds)) initial MPSs")
    #TODO parallelise
    for (di,d) in enumerate(ds)
        opts = _set_options(masteropts; d=d, verbosity=-1)

        sites[di] = siteinds(d, num_mps_sites)
           # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
        Ws[di] = generate_startingMPS(chi_init, sites[di]; num_classes=num_classes, random_state=mps_seed, opts=opts)
    end


    
    #############  initialise the folds  ####################

    nvirt_folds = max(round(Int, ceil(1 / (1-train_ratio); digits=5)), nfolds)
    scv = StratifiedCV(;nfolds=nvirt_folds, rng=foldrng)
    fold_inds = train_test_pairs(scv, eachindex(ys), ys)




    ########### Perform the encoding step for all d and f ############
    #TODO can encode more efficiently for certain encoding types if not time / order dependent. This will save a LOT of memory
    #TODO parallelise
    println("Encoding $nfolds folds with $(length(ds)) different encoding dimensions")
    for f in 1:nfolds, (di,d) in enumerate(ds)
        opts= _set_options(masteropts;  d=d, verbosity=-1)

        tr_inds, val_inds = fold_inds[f]
        local f_Xs_tr = Xs[tr_inds, :]
        local f_Xs_val = Xs[val_inds, :]

        local f_ys_tr = ys[tr_inds]
        local f_ys_val = ys[val_inds]


        range = opts.encoding.range
        scaler = fit_scaler(RobustSigmoidTransform, f_Xs_tr);
        Xs_train_scaled = permutedims(transform_data(scaler, f_Xs_tr; range=range, minmax_output=minmax))
        Xs_val_scaled = permutedims(transform_data(scaler, f_Xs_val; range=range, minmax_output=minmax))


        ########### ATTENTION: due to permutedims, data has been transformed to column major order (each timeseries is a column) #########
        if (isodd(d) && titlecase(encoding.name) == "Sahand") || (d != 2 && titlecase(encoding.name) == "Stoudenmire" )
            continue
        end

        s = EncodeSeparate{opts.encode_classes_separately}()
        training_states, enc_args_tr = encode_dataset(s, Xs_train_scaled, f_ys_tr, "train", sites[di]; opts=opts, class_keys=class_keys)
        validation_states, enc_args_val = encode_dataset(s, Xs_val_scaled, f_ys_val, "valid", sites[di]; opts=opts, class_keys=class_keys)
        
        # enc_args = vcat(enc_args_tr, enc_args_val) 
        Xs_train_enc[di, f] = training_states
        Xs_val_enc[di,f] = validation_states

    end



    #TODO maybe introduce some artificial balancing on the threads, or use a library like transducers
    writelock = ReentrantLock()
    done = Int(sum((!ismissing).(results)) / (max_sweeps+1))
    todo = Int(prod(size(results)) / (max_sweeps+1))
    tstart = time()
    println("Analysing a $todo size parameter grid")
    if distribute
        # the loop order here is: changes execution time the least -> changes execution time the most
        @sync for f in 1:nfolds, (etai, eta) in enumerate(etas), (ei, e) in enumerate(encodings), (di,d) in enumerate(ds), (chmi, chi_max) in enumerate(chi_maxs)
            !ismissing(results[f, 1, etai, di, chmi]) && continue
            isodd(d) && titlecase(e.name) == "Sahand" && continue
            d != 2 && titlecase(e.name) == "Stoudenmire" && continue

            @spawn begin
                opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
                W_init = deepcopy(Ws[di])
                local f_training_states_meta = Xs_train_enc[di, f]
                local f_validation_states_meta = Xs_val_enc[di, f]

                _, info, _, _ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)


                res_by_sweep = Result(info)
                results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value
                lock(writelock)
                try
                    done +=1
                    println("Finished $done/$todo in $(length(res_by_sweep)) sweeps at t=$(time() - tstart)")
                    save_results(resfile, results, f, nfolds, max_sweeps, eta, etas, chi_max, chi_maxs, d, ds, e, encodings) 
                finally
                    unlock(writelock)
                end
            end
        end
    else
        for f in 1:nfolds, (etai, eta) in enumerate(etas), (ei, e) in enumerate(encodings), (di,d) in enumerate(ds), (chmi, chi_max) in enumerate(chi_maxs)
            !ismissing(results[f, 1, etai, di, chmi]) && continue
            isodd(d) && titlecase(e.name) == "Sahand" && continue
            d != 2 && titlecase(e.name) == "Stoudenmire" && continue

            
            opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
            W_init = deepcopy(Ws[di])
            local f_training_states_meta = Xs_train_enc[di, f]
            local f_validation_states_meta = Xs_val_enc[di, f]

            _, info, _, _ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)

            res_by_sweep = Result(info)
            results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value

            done +=1
            println("Finished $done/$todo in $(length(res_by_sweep)) sweeps at t=$(time() - tstart)")
            save_results(resfile, results, f, nfolds, max_sweeps, eta, etas, chi_max, chi_maxs, d, ds, e, encodings) 

        end
    end
    save_status(finfile, nfolds, nfolds, last(etas), etas, last(chi_maxs), chi_maxs, last(ds), ds, last(encodings), encodings)

    return results
end
