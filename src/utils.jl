function load_splits_txt(train_set_location::String, val_set_location::String, 
    test_set_location::String)
    """As per typical UCR formatting, assume labels in first column, followed by data"""
    # do checks
    train_data = readdlm(train_set_location)
    val_data = readdlm(val_set_location)
    test_data = readdlm(test_set_location)

    X_train = train_data[:, 2:end]
    y_train = Int.(train_data[:, 1])

    X_val = val_data[:, 2:end]
    y_val = Int.(val_data[:, 1])

    X_test = test_data[:, 2:end]
    y_test = Int.(test_data[:, 1])

    # recombine val and train into train

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

end

function generate_training_data(samples_per_class::Int, data_pts::Int=5)

    class_A_samples = zeros(samples_per_class, data_pts)
    class_B_samples = ones(samples_per_class, data_pts)
    all_samples = vcat(class_A_samples, class_B_samples)
    all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

    shuffle_idxs = shuffle(1:samples_per_class*2)


    return all_samples[shuffle_idxs, :], all_labels[shuffle_idxs]

end

function generate_sine(n, amplitude=1.0, frequency=1.0)
    t = range(0, 2π, n)
    phase = rand(Uniform(0, 2π)) # randomise the phase
    #amplitude = rand(Uniform(0.1, 1.0))
    return amplitude .* sin.(frequency .* t .+ phase) .+ 0.2 .* randn(n)
end

function generate_gnoise(n, scale=1)
    return randn(n) .* scale
end


function generate_toy_timeseries(time_series_length::Int, total_dataset_size::Int, 
    train_split=0.7; random_state=42, plot_examples=false)
    """Generate two sinusoids of different frequency, and with randomised phase.
    Inject noise with a given amplitude."""
    Random.seed!(random_state)

    train_size = floor(Int, total_dataset_size * train_split)
    test_size = total_dataset_size - train_size

    X_train = zeros(Float64, train_size, time_series_length)
    y_train = zeros(Int, train_size)
    
    X_test = zeros(Float64, test_size, time_series_length)
    y_test = zeros(Int, test_size)

    function generate_sinusoid(length::Int, A::Float64=1.0, 
        f::Float64=1.0, sigma=0.2)
        # sigma is scale of the gaussian noise added to the sinusoid
        t = range(0, 2π, length)
        phase = rand(Uniform(0, 2π)) # randomise the phase

        return A .* sin.(f .*t .+ phase) .+ sigma .* randn(length)

    end

    # generation parameters
    A1, f1, sigma1 = 1.0, 1.0, 0.0 # Class 0
    A2, f2, sigma2 = 1.0, 6.0, 0.0 # Class 1

    for i in 1:train_size
        label = rand(0:1) # choose a label, if 0 use freq f0, if 1 use freq f1. 
        data = label == 0 ? generate_sinusoid(time_series_length, A1, f1, sigma1) : 
            generate_sinusoid(time_series_length, A2, f2, sigma2)
        X_train[i, :] = data
        y_train[i] = label
    end

    for i in 1:test_size
        label = rand(0:1) # choose a label, if 0 use freq f0, if 1 use freq f1. 
        data = label == 0 ? generate_sinusoid(time_series_length, A1, f1, sigma1) : 
            generate_sinusoid(time_series_length, A2, f2, sigma2)
        X_test[i, :] = data
        y_test[i] = label
    end

    # plot some examples
    if plot_examples
        class_0_idxs = findall(x -> x.== 0, y_train)[1:3] # select subset of 5 samples
        class_1_idxs = findall(x -> x.== 1, y_train)[1:3]
        p0 = plot(X_train[class_0_idxs, :]', xlabel="Time", ylabel="x", title="Class 0 Samples (Unscaled)", 
            alpha=0.4, c=:red, label="")
        p1 = plot(X_train[class_1_idxs, :]', xlabel="Time", ylabel="x", title="Class 1 Samples (Unscaled)", 
            alpha=0.4, c=:magenta, label="")
        p = plot(p0, p1, size=(1200, 500), bottom_margin=5mm, left_margin=5mm)
        display(p)
    end

    return (X_train, y_train), (X_test, y_test)

end

function plot_training_summary(info::Dict)
    """Takes in the dictionary of training information 
    and summary information"""
    # extract the keys
    training_loss = info["train_loss"]
    num_sweeps = length(training_loss) - 1
    time_per_sweep = info["time_taken"]

    train_accuracy = info["train_acc"]
    test_accuracy = info["test_acc"]

    train_loss = info["train_loss"]
    test_loss = info["test_loss"]

    # compute the mean time per sweep
    mean_sweep_time = mean(time_per_sweep)
    println("Mean sweep time: $mean_sweep_time (s)")

    # compute the maximum accuracy acheived across any sweep
    max_acc_sweep = argmax(test_accuracy)
    # subtract one because initial test accuracy before training included at index 1
    println("Maximum test accuracy: $(test_accuracy[max_acc_sweep]) achieved on sweep $(max_acc_sweep-1)")

    # create curves
    sweep_range = collect(0:num_sweeps)
    p1 = plot(sweep_range, train_loss, label="train loss", alpha=0.4, c=palette(:default)[1])
    scatter!(sweep_range, train_loss, alpha=0.4, label="", c=palette(:default)[1])
    plot!(sweep_range, test_loss, label="test loss", alpha=0.4, c=palette(:default)[3])
    scatter!(sweep_range, test_loss, alpha=0.4, label="", c=palette(:default)[3])
    xlabel!("Sweep")
    ylabel!("Loss")

    p2 = plot(sweep_range, train_accuracy, label="train acc", c=palette(:default)[1], alpha=0.4)
    scatter!(sweep_range, train_accuracy, label="", c=palette(:default)[1], alpha=0.4)
    plot!(sweep_range, test_accuracy, label="test acc", c=palette(:default)[3], alpha=0.4)
    scatter!(sweep_range, test_accuracy, label="", c=palette(:default)[3], alpha=0.4)
    xlabel!("Sweep")
    ylabel!("Accuracy")

    p3 = bar(collect(1:length(time_per_sweep)), time_per_sweep, label="", color=:skyblue,
        xlabel="Sweep", ylabel="Time taken (s)", title="Training time per sweep")
    
    ps = [p1, p2, p3]

    p = plot(ps..., size=(1000, 500), left_margin=5mm, bottom_margin=5mm)
    display(p)

end


function transform_train_data(X_train::AbstractMatrix; opts::AbstractMPSOptions)
    opts = safe_options(opts) # make sure options is abstract
    # now let's handle the training/testing data
    # rescale using a robust sigmoid transform
    #  Assumes TS are cols 
    

    # transform the data
    # perform the sigmoid scaling
    sig_trans = nothing
    minmax = nothing

    if opts.sigmoid_transform
        sig_trans = Normalization.fit(RobustSigmoid, X_train)

        X_train_scaled = normalize(X_train, sig_trans)
    else
        X_train_scaled = copy(X_train)
    end

    if opts.minmax
        minmax = Normalization.fit(MinMax, X_train_scaled)
        normalize!(X_train_scaled, minmax)

        #TODO introducing the data_bounds parameter broke the nice implementation of normalize.jl :(((, consolidate or go make brendan change his library
        lb, ub = opts.data_bounds
        interval_width = ub - lb

        X_train_scaled .*= interval_width

        X_train_scaled .+= lb
    end


    # map to the domain of the encoding
    a,b = opts.encoding.range
    @. X_train_scaled = (b-a) *X_train_scaled + a
    
    return X_train_scaled, [sig_trans, minmax]
end

function transform_test_data(
        X_test::AbstractMatrix, 
        norms::Vector{<:Union{Nothing, AbstractNormalization}}; 
        opts::AbstractMPSOptions, 
        rescale_out_of_bounds::Bool=true,
        test_print_threshold::Real=0.01
    )
    if isempty(X_test)
        return copy(X_test), []
    end
    opts = safe_options(opts) # make sure options is abstract
    # now let's handle the training/testing data
    # rescale using a robust sigmoid transform
    # Assumes TS are cols
    

    # transform the data
    # perform the sigmoid scaling
    X_test_scaled = copy(X_test)
    for n in norms
        !isnothing(n) && normalize!(X_test_scaled, n)
    end


    if opts.minmax
        
        #TODO introducing the data_bounds parameter broke the nice implementation of normalize.jl :(((, consolidate or go make brendan fix his library
        lb, ub = opts.data_bounds
        interval_width = ub - lb

        X_test_scaled .*= interval_width

        X_test_scaled .+= lb
    end

    oob_rescales = []
    if rescale_out_of_bounds
        # rescale a time-series if out of bounds, this can happen because the minmax scaling of the test set is determined by the train set
        # rescaling like this is undesirable, but allowing time-series to take values outside of [0,1] violates the assumptions of the encoding 
        # and will lead to ill-defined behaviour 
        num_ts_scaled = 0
        for (i, ts) in enumerate(eachcol(X_test_scaled))
            trans = Real[i, 0.,1.] # the "Real" is to stop a conversion to Vector{Float}, as this will cause an indexerror later
            lb, ub = extrema(ts)
            if lb < 0
                if opts.verbosity > 5 && abs(lb) > test_print_threshold
                    @warn "Test series has a value more than 1% below lower bound after train normalization! lb=$lb"
                end
                ts .-= lb
                ub = maximum(ts)
                trans[2] = lb
            end

            if ub > 1
                if opts.verbosity > 5 && abs(ub-1) > test_print_threshold
                    @warn "Test series has a value more than 1% above upper bound after train normalization! ub=$ub"
                end
                ts  ./= ub
                trans[3] = ub
            end
            if !all(trans[2:3] .== [0.,1.])
                num_ts_scaled += 1
                push!(oob_rescales, trans)
            end
        end

        if opts.verbosity > -1 && num_ts_scaled >0
            println("The test set couldn't be perfectly rescaled by the training set normalization, $num_ts_scaled additional rescaling operations had to be performed!")
        end
    end

    # map to the domain of the encoding
    a,b = opts.encoding.range
    @. X_test_scaled = (b-a) *X_test_scaled + a
    
    return X_test_scaled, oob_rescales
end

function transform_test_data(X_test::AbstractVector, args...; kwargs...)

    X_sc_mat, oob_rescales = transform_test_data(reshape(X_test, :,1), args...; kwargs...)
    return X_sc_mat[:], oob_rescales
end


function transform_data(X_train::AbstractMatrix, X_test::AbstractMatrix; opts::AbstractMPSOptions, test_print_threshold::Real=0.01)
    if isempty(X_train) && isempty(X_test)
        return copy(X_train), copy(X_test)
    end
    X_train_scaled, norms = transform_train_data(X_train; opts=opts)
    X_test_scaled, oob_rescales = transform_test_data(X_test, norms; opts=opts, test_print_threshold=test_print_threshold)

    return X_train_scaled, X_test_scaled, norms, oob_rescales 
end



function invert_test_transform(X_test_scaled::Matrix, oob_rescales::AbstractVector, norms::Vector{<:Union{Nothing, AbstractNormalization}}; opts=opts)
    if isempty(X_test_scaled)
        return copy(X_test_scaled), []
    end
    opts = safe_options(opts) # make sure options is abstract
    
    # map back from the domain of the encoding
    a,b = opts.encoding.range
    X_test =  @. ( X_test_scaled - a) / (b-a)

    # reverse any extra out of bounds rescaling
    for (i, lb_shift, ub_scale) in oob_rescales
        @. X_test[:, i] = (X_test[:, i] * ub_scale ) + lb_shift
    end
    
    # undo the effects of data_bounds
    if opts.minmax
        lb, ub = opts.data_bounds
        interval_width = ub - lb

        X_test .-= lb
        X_test ./= interval_width

    end

    # untransform the canonical data transforms
    for n in reverse(norms)
        !isnothing(n) && denormalize!(X_test, n)
    end

    return X_test
end

function invert_test_transform(X_test_scaled::AbstractVector, args...; kwargs...)
    return invert_test_transform(reshape(X_test_scaled, :,1), args...; kwargs...)[:]
end

# itensor deprecated findindex, and firstindex() is broken?
function find_index(t::ITensor, lstr::String) 

    return getfirst(i -> hastags(i, lstr), inds(t))
end

function find_label(W::MPS; lstr="f(x)")
    l_W = lastindex(ITensors.data(W))
    posvec = [l_W, 1:(l_W-1)...]

    for pos in posvec
        label_idx = find_index(W[pos], lstr)
        if !isnothing(label_idx)
            return pos, label_idx
        end
    end
    # @warn "find_label did not find a label index!"
    return nothing, nothing
end

function expand_label_index(mps::MPS; lstr="f(x)")
    "Returns a vector of MPS's, each with a different value set for the label index"

    weights_by_class = []
    pos, l_ind = find_label(mps, lstr=lstr)

    for iv in eachindval(l_ind)
        mpsc = deepcopy(mps)
        mpsc[pos] = mpsc[pos] * onehot(iv)
        normalize!(mpsc)
        push!(weights_by_class, mpsc)
    end
    
    return Vector{MPS}(weights_by_class), l_ind
end



function get_siteinds(W::MPS)
    pos, label_idx = find_label(W)

    if isnothing(pos)
        return siteinds(W)
    else
        W1 = deepcopy(W)
        W1[pos] *= onehot(label_idx => 1) # eliminate label index

        return siteinds(W1)
    end
end

make_windows(ts::Vector{Float64}, w::Int, s::Int) = [ts[i:min(i+w-1, end)] for i in 1:s:length(ts)-w+1]
