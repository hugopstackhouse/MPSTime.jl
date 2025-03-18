"""
    AbstractMPSOptions

Abstract supertype of "MPSOptions", a collection of concrete types which is used to specify options for training, and "Options", which is used internally and contains references to internal objects
"""
abstract type AbstractMPSOptions end

#  New code should use MPSOptions, which is composed of purely concrete types (aside from maybe an abstractRNG object) and won't have the JLD2 serialisation issues


struct MPSOptions <: AbstractMPSOptions
    verbosity::Int # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
    nsweeps::Int # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int # Maximum bond dimension allowed within the MPS during the SVD step
    eta::Float64 # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    d::Int # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
    encoding::Symbol # The type of encoding to use, see structs.jl and encodings.jl for the various options. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
    projected_basis::Bool # whether to pass project=true to the basis
    aux_basis_dim::Int # (NOT IMPLEMENTED) If encoding::SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that num_bins = d / aux_basis_dim. Unused if encoding::Basis
    cutoff::Float64 # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    dtype::DataType # The datatype of the elements of the MPS as well as the encodings. Set to a complex value only if necessary for the encoding type. Supports the arbitrary precsion types BigFloat and Complex{BigFloat}
    loss_grad::Symbol # The type of cost function to use for training the MPS, typically Mean Squared Error or KL Divergence. Must return a vector or pair [cost, dC/dB]
    bbopt::Symbol # Which Black Box optimiser to use, options are Optim or OptimKit derived solvers which work well for MSE costs, or CustomGD, which is a standard gradient descent algorithm with fixed stepsize which seems to give the best results for KLD cost 
    track_cost::Bool # Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers
    rescale::Tuple{Bool,Bool} # Has the form rescale = (before::Bool, after::Bool) and tells the optimisor where to enforce the normalisation of the MPS during training, either calling normalise!(BT) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
    train_classes_separately::Bool # whether the the trainer takes the average MPS loss over all classes or whether it considers each class as a separate problem
    encode_classes_separately::Bool # only relevant for a histogram splitbasis. If true, then the histogram used to determine the bin widths for encoding class A is composed of only data from class A, etc. Functionally, this causes the encoding method to vary depending on the class
    return_encoding_meta_info::Bool # Debug flag: Whether to return the normalised data as well as the histogram bins for the splitbasis types
    minmax::Bool # Whether to apply a minmax norm to the encoded data after it's been SigmoidTransformed
    exit_early::Bool # whether to stop training when train_acc = 1
    sigmoid_transform::Bool # Whether to apply a sigmoid transform to the data before minmaxing
    init_rng::Int # random number generator or seed
    chi_init::Int # Initial bond dimension of the randomMPS
    log_level::Int # 0 for nothing, >0 to save losses, accs, and conf mat. #TODO implement finer grain control
    data_bounds::Tuple{Real, Real} # the region to bound the data to if minmax=true, setting the bounds a bit away from [0,1] can help when the basis is poorly behaved at the boundaries
    use_legacy_ITensor::Bool # Whether to use the old, slow ITensor Implementation
end


"""
    MPSOptions(; <Keyword Arguments>)

Set the hyperparameters and other options for fitMPS. 

# Fields:
## Logging
- `verbosity::Int=1`: How much debug/progress info to print to the terminal while optimising the MPS. Higher numbers mean more output
- `log_level::Int=3`: How much statistical output. 0 for nothing, >0 to print losses, accuracies, and confusion matrix at each step. Noticeable computational overhead 
- `track_cost::Bool=false`: Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers (**HUGE** computational overhead)


## MPS Training Hyperparameters
- `nsweeps::Int=5`: Number of MPS optimisation sweeps to perform (One sweep is both forwards and Backwards)
- `chi_max::Int=25`: Maximum bond dimension allowed within the MPS during the SVD step
- `eta::Float64=0.01`: The learning rate. For gradient descent methods, this is the step size. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
- `d::Int=5`: The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
- `cutoff::Float64=1E-10`: Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
- `dtype::DataType=Float64 or ComplexF64 depending on encoding`: The datatype of the elements of the MPS. Supports the arbitrary precsion types such as BigFloat and Complex{BigFloat}
- `exit_early::Bool=false`: Stops training if training accuracy is 1 at the end of any sweep.


## Encoding Options
- `encoding::Symbol=:Legendre`: The encoding to use, including :Stoudenmire, :Fourier, :Legendre, :SLTD, :Custom, etc. see Encoding docs for a complete list. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
- `projected_basis::Bool=false`: Whether toproject a basis onto the training data at each time. Normally, when specifying a basis of dimension *d*, the first *d* lowest order terms are used. When project=true, the training data is used to construct a pdf of the possible timeseries amplitudes at each time point. The first *d* largest terms of this pdf expanded in a series are used to select the basis terms.
- `aux_basis_dim::Int=2`: Unused for standard encodings. If the encoding is a SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that the number of histogram bins = *d* / *aux_basis_dim*. 
- `encode_classes_separately::Bool=false`: Only relevant for data driven bases. If true, then data is split up by class before being encoded. Functionally, this causes the encoding method to vary depending on the class


## Data Preprocessing and MPS initialisation
- `sigmoid_transform::Bool`: Whether to apply a sigmoid transform to the data before minmaxing. This has the form
```math
\\boldsymbol{X'} = \\left(1 + \\exp{-\\frac{\\boldsymbol{X}-m_{\\boldsymbol{X}}}{r_{\\boldsymbol{X}} / 1.35}}\\right)^{-1}
```

where ``\\boldsymbol{X}`` is the un-normalized time-series data matrix, ``m_{\\boldsymbol{X}}`` is the median of ``\\boldsymbol{X}`` and ``r_{\\boldsymbol{X}}``is its interquartile range.

- `minmax::Bool`: Whether to apply a minmax norm to `[0,1]` before encoding. This has the form
```math
\\boldsymbol{X'} =  \\frac{\\boldsymbol{X} - x'_{\\text{min}}}{x'_{\\text{max}} - x'_{\\text{min}}},
```

where ``\\boldsymbol{X''}`` is the scaled robust-sigmoid transformed data matrix, ``x'_\\text{min}`` and ``x'_\\text{max}`` are the minimum and maximum of ``\\boldsymbol{X'}``.
- `data_bounds::Tuple{Float64, Float64} = (0.,1.)`: The region to bound the data to if minmax=true. This is separate from the encoding domain. All encodings expect data to be scaled scaled between 0 and 1. Setting the data bounds a bit away from [0,1] can help when your basis has poor support near its boundaries.

- `init_rng::Int`: Random seed used to generate the initial MPS
- `chi_init::Int`: Initial bond dimension of the random MPS


## Loss Functions and Optimisation Methods
- `loss_grad::Symbol=:KLD`: The type of cost function to use for training the MPS, typically Mean Squared Error (:MSE) or KL Divergence (:KLD), but can also be a weighted sum of the two (:Mixed) if use_legacy_ITensor is enabled.
- `bbopt::Symbol=:TSGO`: Which local Optimiser to use, builtin options are symbol gradient descent (:GD), or gradient descent with a TSGO rule (:TSGO). If `use_legacy_ITensor`` is enabled, can be a Conjugate Gradient descent optimisation rule using either the Optim or OptimKit package (:Optim or :OptimKit respectively). The CGD methods work well for MSE based loss functions, but seem to perform poorly for KLD base loss functions.

- `rescale::Tuple{Bool,Bool}=(false,true)`: Has the form `rescale = (before::Bool, after::Bool)`. Where to enforce the normalisation of the MPS during training, either calling normalise!(*Bond Tensor*) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
- `update_iters::Int=1`: Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by TSGO, Optim or OptimKit
- `train_classes_separately::Bool=false`: Whether the the trainer optimises the total MPS loss over all classes or whether it considers each class as a separate problem. Should make very little diffence
- `use_legacy_ITensor::Bool=false`: Whether to use the old, slow ITensor Implementation


## Debug
- `return_encoding_meta_info::Bool=false`: Debug flag: Whether to return the normalised data as well as the histogram bins for the splitbasis types

"""
function MPSOptions(;
    verbosity::Int=1, # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
    nsweeps::Int=5, # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int=25, # Maximum bond dimension allowed within the MPS during the SVD step
    eta::Float64=0.01, # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    d::Int=5, # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
    encoding::Symbol=:Legendre_No_Norm, # The type of encoding to use, see structs.jl and encodings.jl for the various options. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
    projected_basis::Bool=false, # whether to pass project=true to the basis
    aux_basis_dim::Int=2, # (NOT IMPLEMENTED) If encoding::SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that num_bins = d / aux_basis_dim. Unused if encoding::Basis
    cutoff::Float64=1E-10, # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int=1, # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    dtype::DataType=(model_encoding(encoding).iscomplex ? ComplexF64 : Float64), # The datatype of the elements of the MPS as well as the encodings. Set to a complex value only if necessary for the encoding type. Supports the arbitrary precsion types BigFloat and Complex{BigFloat}
    loss_grad::Symbol=:KLD, # The type of cost function to use for training the MPS, typically Mean Squared Error or KL Divergence. Must return a vector or pair [cost, dC/dB]
    bbopt::Symbol=:TSGO, # Which Black Box optimiser to use, options are Optim or OptimKit derived solvers which work well for MSE costs, or CustomGD, which is a standard gradient descent algorithm with fixed stepsize which seems to give the best results for KLD cost 
    track_cost::Bool=false, # Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers
    rescale::Tuple{Bool,Bool}=(false, true), # Has the form rescale = (before::Bool, after::Bool) and tells the optimisor where to enforce the normalisation of the MPS during training, either calling normalise!(BT) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
    train_classes_separately::Bool=false, # whether the the trainer takes the average MPS loss over all classes or whether it considers each class as a separate problem
    encode_classes_separately::Bool=false, # only relevant for a histogram splitbasis. If true, then the histogram used to determine the bin widths for encoding class A is composed of only data from class A, etc. Functionally, this causes the encoding method to vary depending on the class
    return_encoding_meta_info::Bool=false, # Debug flag: Whether to return the normalised data as well as the histogram bins for the splitbasis types
    minmax::Bool=true, # Whether to apply a minmax norm to the encoded data after it's been SigmoidTransformed
    exit_early::Bool=false, # whether to stop training when train_acc = 1
    sigmoid_transform::Bool=true, # Whether to apply a sigmoid transform to the data before minmaxing
    init_rng::Int=1234, # SEED ONLY IMPLEMENTED (Itensors fault) random number generator or seed Can be manually overridden by calling fitMPS(...; random_seed=val)
    chi_init::Int=4, # Initial bond dimension of the randomMPS fitMPS(...; chi_init=val)
    log_level::Int=3, # 0 for nothing, >0 to save losses, accs, and conf mat. #TODO implement finer grain control
    data_bounds::Tuple{Real, Real}=(0.,1.),
    use_legacy_ITensor::Bool=true # Whether to use the old, slow ITensor Implementation
    )

    return MPSOptions(verbosity, nsweeps, chi_max, eta, d, encoding, 
        projected_basis, aux_basis_dim, cutoff, update_iters, 
        dtype, loss_grad, bbopt, track_cost, rescale, 
        train_classes_separately, encode_classes_separately, 
        return_encoding_meta_info, minmax, exit_early, 
        sigmoid_transform, init_rng, chi_init, log_level, data_bounds,use_legacy_ITensor
    )
end




# container for options with default values
"""
    Options(; <Keyword Arguments>)

The internal options struct. Fields have the same meaning as MPSOptions, but contains objects instead of symbols, e.g. Encoding=Basis("Legendre") instead of :Legendre
"""
struct Options <: AbstractMPSOptions
    verbosity::Int # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
    nsweeps::Int # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int # Maximum bond dimension allowed within the MPS during the SVD step
    cutoff::Float64 # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    dtype::DataType # The datatype of the elements of the MPS as well as the encodings. Set to a complex value only if necessary for the encoding type. Supports the arbitrary precsion types BigFloat and Complex{BigFloat}
    loss_grad::Function # The type of cost function to use for training the MPS, typically Mean Squared Error or KL Divergence. Must return a vector or pair [cost, dC/dB]
    bbopt::BBOpt # Which Black Box optimiser to use, options are Optim or OptimKit derived solvers which work well for MSE costs, or CustomGD, which is a standard gradient descent algorithm with fixed stepsize which seems to give the best results for KLD cost 
    track_cost::Bool # Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers
    eta::Float64 # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    rescale::Tuple{Bool,Bool} # Has the form rescale = (before::Bool, after::Bool) and tells the optimisor where to enforce the normalisation of the MPS during training, either calling normalise!(BT) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
    d::Int # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
    aux_basis_dim::Int # If encoding::SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that num_bins = d / aux_basis_dim. Unused if encoding::Basis
    encoding::Encoding # The type of encoding to use, see structs.jl and encodings.jl for the various options. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
    train_classes_separately::Bool # whether the the trainer takes the average MPS loss over all classes or whether it considers each class as a separate problem
    encode_classes_separately::Bool # only relevant for a histogram splitbasis. If true, then the histogram used to determine the bin widths for encoding class A is composed of only data from class A, etc. Functionally, this causes the encoding method to vary depending on the class
    #allow_unsorted_class_labels::Bool #Notimplemeted Allows the class labels to be unsortable types. This does not affect the training in anyway, but will lead to oddly ordered output in the summary statistics
    return_encoding_meta_info::Bool # Debug flag: Whether to return the normalised data as well as the histogram bins for the splitbasis types
    minmax::Bool # Whether to apply a minmax norm to the encoded data after it's been SigmoidTransformed
    exit_early::Bool # whether to stop training when train_acc = 1
    sigmoid_transform::Bool # Whether to apply a sigmoid transform to the data before minmaxing
    log_level::Int # 0 for nothing, >0 to save losses, accs, and conf mat. #TODO implement finer grain control
    data_bounds::Tuple{<:Real, <:Real} # the region to bound the data to if minmax=true, setting the bounds a bit away from [0,1] can help when the basis is poorly behaved at the boundaries
    chi_init::Int # initial bond dimension of the mps (before any optimisation)
    init_rng::Int # initial rng seed for generating the MPS
    use_legacy_ITensor::Bool # Whether to use the old, slow ITensor Implementation
end

function Options(; 
        nsweeps=5, 
        chi_max=25, 
        cutoff=1E-10, 
        update_iters=1, 
        verbosity=1, 
        loss_grad=loss_grad_KLD, 
        bbopt=BBOpt("CustomGD", "TSGO"),
        track_cost::Bool=(verbosity >=1), 
        eta=0.01, rescale = (false, true), 
        d=5, 
        aux_basis_dim=1, 
        encoding=legendre_no_norm(), 
        dtype::DataType=encoding.iscomplex ? ComplexF64 : Float64, 
        train_classes_separately::Bool=false, 
        encode_classes_separately::Bool=train_classes_separately, 
        return_encoding_meta_info=false, 
        minmax=true, 
        exit_early=true, 
        sigmoid_transform=true, 
        log_level=3, 
        projected_basis=false,
        data_bounds::Tuple{<:Real, <:Real}=(0.,1.),
        chi_init::Integer=4,
        init_rng::Integer=1234,
        use_legacy_ITensor::Bool=true # Whether to use the old, slow ITensor Implementation
    )

    if encoding isa Symbol
        encoding = model_encoding(encoding, projected_basis)
    end

    if bbopt isa Symbol
        bbopt = model_bbopt(bbopt)
    end

    if loss_grad isa Symbol 
        loss_grad = model_loss_func(loss_grad)
    end

    Options(verbosity, nsweeps, chi_max, cutoff, update_iters, 
        dtype, loss_grad, bbopt, track_cost, 
        eta, rescale, d, aux_basis_dim, encoding, train_classes_separately, 
        encode_classes_separately, return_encoding_meta_info, 
        minmax, exit_early, sigmoid_transform, log_level, data_bounds, 
        chi_init, init_rng, use_legacy_ITensor
        )

end

"""
    model_encoding(symb::Symbol, project::Bool=false)

Construct an Encoding object from *symb*. Not case sensitive. See Encodings documentation for the full list of options. Will use the specified project options if the encoding supports projecting. The inverse of symbolic_encoding.

"""
function model_encoding(symb::Symbol, project::Bool=false)
    s = symb |> String |> lowercase
    if s in ["legendre_no_norm", "legendre"]
        enc = legendre_no_norm(project=project)
    elseif s == "legendre_norm"
        enc = legendre(project=project, norm=true)
    elseif s == "stoudenmire"
        enc = stoudenmire()
    elseif s == "fourier"
        enc = fourier(project=project)
    elseif s == "sahand"
        enc = sahand()
    elseif s in ["sl", "sahand_legendre", "sahand_legendre_time_independent", "sahand-legendre_time_independent"]
        enc = sahand_legendre(false)
    elseif s in ["sltd", "sahand_legendre_time_dependent", "sahand-_legendre_time_dependent"]
        enc = sahand_legendre(true)
    elseif s == "uniform"
        enc = uniform()
    elseif startswith(s, "hist_split_") || startswith(s, "hist._split_") || startswith(s, "histogram_split_")
        i = 6
        while s[i-5:i] !== "split_"
            i += 1
        end
        enc = histogram_split(model_encoding(Symbol(s[i+1:end])))
    elseif startswith(s, "unif_split_") || startswith(s, "unif._split_") || startswith(s, "uniform_split_")
        i = 6
        while s[i-5:i] !== "split_"
            i += 1
        end
        enc = uniform_split(model_encoding(Symbol(s[i+1:end])))
    elseif s  == "custom"
        enc = erf()  # placeholder
    else
        throw(ArgumentError("Unknown encoding function \"$s\". Please use one of :Legendre, Stoudenmire, :Fourier, Sahand_Legendre, :Custom etc."))
    end
    return enc
end

"""
    symbolic_encoding(E::Encoding)

Construct a symbolic name from an Encoding object. The inverse of model_encoding
"""
function symbolic_encoding(E::Encoding)
    str = E.name
    return Symbol(replace(str," " => "_", "-" => "_"))
end



"""
    model_bbopt(symb::Symbol)

Constuct a BBOpt object from *symb*. Not case sensitive.
"""
function model_bbopt(symb::Symbol)
    s = symb |> String |> lowercase
    if s in ["gd", "customgd"]
        opt = BBOpt("CustomGD")
    elseif s == "tsgo"
        opt = BBOpt("CustomGD", "TSGO")
    elseif s == "optim"
        opt = BBOpt("Optim")
    elseif s == "optimkit"
        opt = BBopt("OptimKit")
    end

    return opt
end

"""
    model_loss_func(symb::Symbol)

Select a loss function (::Function) from the *symb*. Not case sensitive. The inverse of *symbolic_loss_func*
"""
function model_loss_func(s::Symbol)
    if s == :KLD
        lf = loss_grad_KLD
    elseif s == :MSE
        lf = loss_grad_MSE
    elseif s == :Mixed
        lf = loss_grad_mixed
    end
    return lf
end

function symbolic_loss_func(f::Function)
    if f == loss_grad_KLD
        s =  :KLD
    elseif f == loss_grad_MSE
        s = :MSE
    elseif f == loss_grad_mixed 
        s =  :Mixed
    end
    return s
end

"""Convert a serialisable MPSOptions into the internal Options type."""
function Options(m::MPSOptions)
    properties = propertynames(m)

    # this is actually cool syntax I have to say
    opts = Options(; [field => getfield(m,field) for field in properties]...)
    return opts

end

"""Convert the internal Options type into a serialisable MPSOptions."""
function MPSOptions(opts::Options,)
    properties = propertynames(opts)
    properties = filter(s -> !(s in [:encoding, :bbopt, :loss_grad]), properties)

    sencoding = symbolic_encoding(opts.encoding)
    sbbopt = Symbol(opts.bbopt.fl)
    sloss_grad = symbolic_loss_func(opts.loss_grad)

    mopts = MPSOptions(; [field => getfield(opts,field) for field in properties]..., encoding=sencoding, bbopt=sbbopt, loss_grad=sloss_grad)
    return mopts
end

# ability to "modify" options. Useful in very specific cases.
function _set_options(opts::AbstractMPSOptions; kwargs...)
    properties = propertynames(opts)
    kwkeys = keys(kwargs)
    bad_key = findfirst( map((!key -> hasfield(typeof(opts), key)), kwkeys))

    if !isnothing(bad_key)
        throw(ErrorException("type $typeof(opts) has no field $(kwkeys[bad_key])"))
    end
    
    # this is actually cool syntax I have to say
    return typeof(opts)(; [field => getfield(opts,field) for field in properties]..., kwargs...)
end

Options(opts::Options; kwargs...) = _set_options(opts; kwargs...)

function default_iter()
    @error("No loss_gradient function defined in options")
end


"""
    safe_options(opts::AbstractMPSOptions)
    
Takes any AbstractMPSOptions type, and returns an instantiated Options type.
"""
function safe_options(opts::MPSOptions)

    abs_opts = Options(opts)
    if opts.verbosity >=5
        println("converting MPSOptions to concrete Options object")
    end

    return abs_opts
end

safe_options(options::Options) = options


# 
"""
    TrainedMPS

Container for a trained MPS and its associated Options and training data.

# Fields
- `mps::MPS`: A trained Matrix Product state.
- `opts::MPSOptions`: User defined MPSOptions used to create the MPS.
- `train_data::EncodedTimeSeriesSet`: Stores both the raw and encoded data used to train the mps.
"""
struct TrainedMPS
    mps::MPS
    opts::MPSOptions
    # opts_concrete::Options # BAD for serialisation
    train_data::EncodedTimeSeriesSet
end
