################## Splitting Initialisers
function get_nbins_safely(opts)
    nbins = opts.d / opts.aux_basis_dim
    if opts.d % opts.aux_basis_dim !== 0
        throw(ArgumentError("The auxilliary basis dimension ($(opts.aux_basis_dim)) must evenly divide the total feature dimension ($(opts.d))"))
    end

    return convert(Int, nbins) # try blocks have their own scope
end


function split_init(X_norm::AbstractMatrix, y::AbstractVector; opts::Options, range::Tuple{Real, Real}=opts.encoding.range)
    nbins = get_nbins_safely(opts)

    bins = opts.encoding.splitmethod(X_norm, nbins, range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.aux_enc]

    if opts.encoding.aux_enc isa SplitBasis 
        # This loop is to handle deciding the aux_dim of nested split bases. #TODO make aux_dim a parameter of split_basis instead
        aux_aux_dim = 1
        i = 2
        while opts.aux_basis_dim > i
            if opts.aux_basis_dim % i == 0
                aux_aux_dim = i
            end
            i += 1
        end
        aux_opts=_set_options(opts; encoding = opts.encoding.aux_enc, d=opts.aux_basis_dim, aux_basis_dim=aux_aux_dim)
    else
        aux_opts=_set_options(opts; encoding = opts.encoding.aux_enc, d=opts.aux_basis_dim)

    end        

    if opts.encoding.aux_enc.isdatadriven #TODO, implement this properly, currently forbidden
        if eltype(bins) <: Number
            aux_enc_args = [opts.encoding.aux_enc.init(X_norm, y; opts=aux_opts, range=(bins[i], bins[i+1])) for i in 1:nbins]

        else
            aux_enc_args = [[opts.encoding.aux_enc.init(X_norm, y; opts=aux_opts, range=(bins[ti][i], bins[ti][i+1])) for i in 1:nbins] for ti in eachindex(bins) ]
        end
    else
        enc_arg = opts.encoding.aux_enc.init(X_norm, y; opts=aux_opts)
        aux_enc_args = [enc_arg for _ in 1:nbins]
    end


    return [aux_enc_args, split_args]
end

##################### Splitting methods
function unif_split(data::AbstractMatrix, nbins::Integer, a::Real, b::Real)
    dx = (b-a)/nbins# width of one interval
    return collect(a:dx:b)
end

function hist_split(samples::AbstractVector, nbins::Integer, a::Real, b::Real) # samples should be a vector of timepoints at a single time (NOT a timeseries) for all sensible use cases
    npts = length(samples)
    bin_pts = Int(round(npts/nbins))

    if bin_pts == 0
        @warn("Less than one data point per bin! Putting the extra bins at x=1 and hoping for the best")
        bin_pts = 1
    end

    bins = fill(convert(eltype(samples), a), nbins+1) # look I'm not happy about this syntax either. Why does zeros() take a type, but not fill()?
    
    #bins[1] = a # lower bound
    j = 2
    ds = sort(samples[@. a <=  samples <= b]) # only consider samples between a and b, this makes nested splitbases work
    for (i,x) in enumerate(ds)
        if i % bin_pts == 0 && i < length(samples)
            if j == nbins + 1
                # This can happen if bin_pts is very small due to a small dataset, e.g. npts = 18, nbins = 8, then we can get as high as j = 10 and IndexError!
                #@warn("Only $bin_pts data point(s) per bin! This may seriously bias the encoding/lead to per performance (last bin contains $(npts - i) extra points)")
                break
            end
            bins[j] = (x + ds[i+1])/2
            j += 1
        end
    end
    if j <=  nbins
        bins[bins .== a] .= b 
        bins[1] = a
    end

    bins[end] = b # upper bound
    return bins
end

function hist_split(X_norm::AbstractMatrix, nbins::Integer, a::Real, b::Real)
    return [hist_split(samples,nbins, a, b) for samples in eachrow(X_norm)]
end


################## Splitting encoding helpers
function rect(x::Real, lbound::Real=0.5, rbound::Real=0.5)
    # helper used to construct the split basis. It is important that rect(0.5) returns 0.5 because if an encoded point lies exactly on a bin boundary we want enc(x) = (0,...,0, 0.5, 0.5, 0,...0)
    # (Having two 1s instead of two 0.5s would violate our normalised encoding assumption)
    if x == -0.5
        return lbound
    elseif x == 0.5
        return rbound
    elseif (-0.5 <= x <= 0.5)
        return 1.
    else
        return 0.
    end
end

# x: timepoints
# aux_vec: function that takes x -> aux_encoding(x)
# bins: the bin edges
function project_onto_bins(x::Float64, aux_dim::Int, aux_encoder::Function, bins::AbstractVector; norm::Bool=true)
    widths = diff(bins)
    a,b = bins[1], bins[end] # there will always be at least two bin edges 
    scale = b-a

    encoding = []
    for (i, dx) in enumerate(widths)

        # The aux basis is orthonormal on [a,b] and is stretched onto a bin of width dx, so sqrt(scale/dx) keeps it orthonormal on the bin
        y = norm ? sqrt(scale/dx) : 1/dx
        lbound = i == 1 ? 1. : 0.5
        rbound = i == length(widths) ? 1. : 0.5
        x_prop = scale*(x - bins[i]) / dx
        select = y * rect(x_prop/scale - 0.5, lbound, rbound )
        aux_encoding = select == 0 ? zeros(aux_dim) : select .* aux_encoder(a+x_prop, i) # short circuit eval is necessary so that we don't evaluate aux enc function out of its domain
        push!(encoding, aux_encoding)
    end


    return vcat(encoding...)
end


function project_onto_bins(x::Float64, d::Int, aux_enc_args::AbstractVector, split_args::AbstractVector; norm::Bool=true)
    bins::AbstractVector{Float64}, aux_dim::Int, aux_enc::Encoding = split_args

    aux_encoder = (xx, bin_num) -> aux_enc.encode(xx, aux_dim, aux_enc_args[bin_num]...)

    return project_onto_bins(x, aux_dim, aux_encoder, bins; norm=norm)

end

function project_onto_bins(x::Float64, d::Int, ti::Int, all_aux_enc_args::AbstractVector, split_args::AbstractVector; norm::Bool=true)
    all_bins::AbstractVector, aux_dim::Int, aux_enc::Encoding = split_args

    if eltype(all_bins) <: Number
        bins = all_bins
    else
        bins = all_bins[ti]
    end

    if aux_enc.istimedependent
        aux_enc_args = all_aux_enc_args[ti]
        aux_encoder = (xx, bin_num) -> aux_enc.encode(xx, aux_dim, ti, aux_enc_args[bin_num]...)
    else
        aux_enc_args = all_aux_enc_args
        aux_encoder = (xx, bin_num) -> aux_enc.encode(xx, aux_dim, aux_enc_args[bin_num]...)
    end

    return project_onto_bins(x, aux_dim, aux_encoder, bins; norm=norm)

end
