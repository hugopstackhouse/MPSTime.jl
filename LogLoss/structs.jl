using Parameters
using ITensors

import Base.convert
# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = AbstractVector{ITensor} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 

# value types
struct TrainSeparate{Bool} end # value type to determine whether training is together or separate
struct EncodeSeparate{Bool} end # value type for dispatching on whether to encode classes separately


# data structures
struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    label_index::UInt
end
const TimeseriesIterable = Vector{PState}
struct EncodedTimeseriesSet
    timeseries::TimeseriesIterable
    class_distribution::Vector{Integer}
end



# Black box optimiser shell
struct BBOpt 
    name::String
    fl::String
    BBOpt(s::String, fl::String) = begin
        if !(lowercase(s) in ["optim", "optimkit", "customgd"]) 
            error("Unknown Black Box Optimiser $s, options are [CustomGD, Optim, OptimKit]")
        end
        new(s,fl)
    end
end

function BBOpt(s::String)
    sl = lowercase(s)
    if sl == "customgd"
        return BBOpt(s, "GD")
    else
        return BBOpt(s, "CGD")
    end
end
function Base.show(io::IO, O::BBOpt)
    print(io,O.name," with ", O.fl)
end

include("basis_structs.jl")

# container for options with default values

function default_iter()
    @error("No loss_gradient function defined in options")
end
@with_kw struct Options
    nsweeps::Int # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int # Maximum bond dimension allowed within the MPS during the SVD step
    cutoff::Float64 # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    verbosity::Int # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
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
end

function Options(; nsweeps=5, chi_max=25, cutoff=1E-10, update_iters=10, verbosity=1, dtype::DataType=ComplexF64, loss_grad=loss_grad_KLD, bbopt=BBOpt("CustomGD"),
    track_cost::Bool=(verbosity >=1), eta=0.01, rescale = (false, true), d=2, aux_basis_dim=1, encoding=Basis("Stoudenmire"), train_classes_separately::Bool=false, 
    encode_classes_separately::Bool=train_classes_separately, return_encoding_meta_info=false, minmax=true)
    Options(nsweeps, chi_max, cutoff, update_iters, verbosity, dtype, loss_grad, bbopt, track_cost, eta, rescale, d, aux_basis_dim, encoding, train_classes_separately, 
        encode_classes_separately, return_encoding_meta_info, minmax)
end


# ability to modify options 
function _set_options(opts::Options; kwargs...)
    properties = propertynames(opts)
    kwkeys = keys(kwargs)
    bad_key = findfirst( map((!key -> hasfield(Options, key)), kwkeys))

    if !isnothing(bad_key)
        throw(ErrorException("type Options has no field $(kwkeys[bad_key])"))
    end

    # this is actually cool syntax I have to say
    return Options(; [field => getfield(opts,field) for field in properties]..., kwargs...)

end

Options(opts::Options; kwargs...) = _set_options(opts; kwargs...)


# type conversions
# These are reasonable to implement because Basis() and BBOpt() are just wrapper types with some validation built in
convert(::Type{Basis}, s::String) = Basis(s)
convert(::Type{BBOpt}, s::String) = BBOpt(s)