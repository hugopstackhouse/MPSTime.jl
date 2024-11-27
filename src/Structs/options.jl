
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
    data_bounds::Tuple{Float64, Float64} # the region to bound the data to if minmax=true, setting the bounds a bit away from [0,1] can help when the basis is poorly behaved at the boundaries
end

function MPSOptions(;
    verbosity::Int=1, # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
    nsweeps::Int=5, # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int=15, # Maximum bond dimension allowed within the MPS during the SVD step
    eta::Float64=0.01, # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    d::Int=2, # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
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
    exit_early::Bool=true, # whether to stop training when train_acc = 1
    sigmoid_transform::Bool=true, # Whether to apply a sigmoid transform to the data before minmaxing
    init_rng::Int=1234, # SEED ONLY IMPLEMENTED (Itensors fault) random number generator or seed Can be manually overridden by calling fitMPS(...; random_seed=val)
    chi_init::Int=4, # Initial bond dimension of the randomMPS fitMPS(...; chi_init=val)
    log_level::Int=3, # 0 for nothing, >0 to save losses, accs, and conf mat. #TODO implement finer grain control
    data_bounds::Tuple{Float64, Float64}=(0.,1.)
    )

    return MPSOptions(verbosity, nsweeps, chi_max, eta, d, encoding, 
        projected_basis, aux_basis_dim, cutoff, update_iters, 
        dtype, loss_grad, bbopt, track_cost, rescale, 
        train_classes_separately, encode_classes_separately, 
        return_encoding_meta_info, minmax, exit_early, 
        sigmoid_transform, init_rng, chi_init, log_level, data_bounds)
end




# container for options with default values

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
    data_bounds::Tuple{Float64, Float64} # the region to bound the data to if minmax=true, setting the bounds a bit away from [0,1] can help when the basis is poorly behaved at the boundaries
    chi_init::Int # initial bond dimension of the mps (before any optimisation)
    init_rng::Int # initial rng seed for generating the MPS
end

function Options(; 
        nsweeps=5, 
        chi_max=25, 
        cutoff=1E-10, 
        update_iters=10, 
        verbosity=1, 
        loss_grad=loss_grad_KLD, 
        bbopt=BBOpt("CustomGD"),
        track_cost::Bool=(verbosity >=1), 
        eta=0.01, rescale = (false, true), 
        d=2, 
        aux_basis_dim=1, 
        encoding=stoudenmire(), 
        dtype::DataType=encoding.iscomplex ? ComplexF64 : Float64, 
        train_classes_separately::Bool=false, 
        encode_classes_separately::Bool=train_classes_separately, 
        return_encoding_meta_info=false, 
        minmax=true, 
        exit_early=true, 
        sigmoid_transform=true, 
        log_level=3, 
        projected_basis=false,
        data_bounds::Tuple{Float64, Float64}=(0.,1.),
        chi_init::Integer=4,
        init_rng::Integer=1234
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
        chi_init, init_rng
        )

end

function model_encoding(s::Symbol, proj::Bool=false)
    if s in [:Legendre_No_Norm, :legendre_no_norm, :Legendre, :legendre]
        enc = legendre_no_norm(project=proj)
    elseif s in [:Legendre_Norm, :legendre_norm]
        enc = legendre(project=proj)
    elseif s in [:Stoudenmire, :stoudenmire]
        enc = stoudenmire()
    elseif s in [:Fourier, :fourier]
        enc = fourier(project=proj)
    elseif s in [:Sahand, :sahand]
        enc = sahand()
    elseif s in [:SL, :Sahand_Legendre, :sahand_legendre]
        enc = sahand_legendre(false)
    elseif s in [:SLTD, :Sahand_Legendre_Time_Dependent, :sahand_legendre_time_dependent]
        enc = sahand_legendre(true)
    elseif s in [:Uniform, :uniform]
        enc = uniform()
    elseif s in [:Custom, :custom]
        enc = erf()  # placeholder
    else
        raise(ArgumentError("Unknown encoding function. Please use one of :Legendre, Stoudenmire, :Fourier, Sahand_Legendre, etc."))
    end
    return enc
end

function symbolic_encoding(E::Encoding)
    str = E.name
    return Symbol(replace(str," " => "_"))
end

function model_bbopt(s::Symbol)
    if s in [:GD, :CustomGD]
        opt = BBOpt("CustomGD")
    elseif s == :TSGO
        opt = BBOpt("CustomGD", "TSGO")
    elseif s == :Optim
        opt = BBOpt("Optim")
    elseif s == :OptimKit
        opt = BBopt("OptimKit")
    end

    return opt
end


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

"""Convert the concrete MPSOpts to the abstract Options type that is needed for runtime but doesn't serialize as well"""
function Options(m::MPSOptions)
    properties = propertynames(m)

    # this is actually cool syntax I have to say
    opts = Options(; [field => getfield(m,field) for field in properties]...)
    return opts

end

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



function safe_options(opts::MPSOptions)

    abs_opts = Options(opts)
    if opts.verbosity >=5
        println("converting MPSOptions to concrete Options object")
    end

    return abs_opts
end

safe_options(options::Options) = options


# container for a trained MPS, options, and training data
struct TrainedMPS
    mps::MPS
    opts::MPSOptions
    opts_concrete::Options
    train_data::EncodedTimeseriesSet
end
