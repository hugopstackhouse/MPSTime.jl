
function plot_encoding(
    E::Encoding, 
    d::Integer, 
    X_train::Matrix{Float64} = zeros(0,0), 
    y_train::Vector{Any}=[];
    tis::Vector{<:Integer} = Int[],
    ds::Vector{<:Integer} = collect(1:d),
    minmax::Bool=true, 
    sigmoid_transform::Bool=false,
    data_bounds::Tuple{<:Real, <:Real}=(0.,1.),
    num_xvals::Integer=500,
    plot_hist::Bool=E.isdatadriven,
    size::Tuple=(1200, 800),
    padding::Real=6.,
    aux_basis_dim::Integer=2,
    return_subplots=false,
    kwargs... # passed to plot
    )

    # purely for the sake of norming the train data
    opts = Options(; encoding=E, minmax=minmax, d=d, sigmoid_transform=sigmoid_transform, data_bounds=data_bounds, aux_basis_dim=aux_basis_dim)

    if E.isdatadriven || E.istimedependent
        if isempty(X_train)
            throw(ArgumentError("""The encoding \"$(B.name)\" is data driven or time dependent, please call plot_encoding with the signature:

                plot_encoding(E::Encoding, d::Integer, X_train::Matrix{Float64}, y_train::Vector...; minmax::Bool, sigmoid_transform::Bool, data_bounds::Tuple,...)

            (Unless you've made a custom basis that uses it, you don't actually need y_train)"""))
        end
        X_norm, _... = transform_train_data(permutedims(X_train); opts)
        encoding_args = opts.encoding.init(X_norm, y_train; opts=opts)

    elseif plot_hist
        if isempty(X_train)
            throw(ArgumentError(""" Cannot plot data histogram without the X_train parameter being passed, call plot_encoding with the signature:

                plot_encoding(E::Encoding, d::Integer, X_train::Matrix{Float64},...; minmax::Bool, sigmoid_transform::Bool, data_bounds::Tuple,...)
            """))
        end
        X_norm, _... = transform_train_data(permutedims(X_train); opts)
        encoding_args = E.init(X_norm, y_train; opts=opts)
    else
        X_norm = zeros(0,0)
        encoding_args = E.init(X_train, y_train; opts=opts)
    end

    if isempty(tis)
        tis = [1]
    end

    basis_per_time = []
    xvals = range(opts.encoding.range..., num_xvals)

    for time in tis
        basis = Matrix{Number}(undef, num_xvals, d)
        if E.istimedependent
            for (i,x) in enumerate(xvals)
                basis[i, :] = opts.encoding.encode(x, opts.d, time, encoding_args...)
            end
        else
            for (i,x) in enumerate(xvals)
                basis[i, :] = opts.encoding.encode(x, opts.d, encoding_args...)
            end
        end
        push!(basis_per_time, basis)
    end


    ps = []
    for i in eachindex(basis_per_time)
        basis = basis_per_time[i]
        t = tis[i]

        p = plot()

        if E.iscomplex 
            for l in ds
                p = plot(p, xvals, real.(basis[:, l]); label=L"\textrm{Real}\{b_{%$(l)}\}")
                p = plot(p, xvals, imag.(basis[:, l]); label=L"\textrm{Imag}\{b_{%$(l)}\}")
            end
        else
            for l in ds
                p = plot(p, xvals, basis[:, l]; label=L"b_{%$(l)}")
            end

        end

        xlabel!("x")
        ylabel!("y")
        title!("$(E.name) Encoding\nat t=$t")

        if plot_hist
            if E.istimedependent
                h = histogram(X_norm[t,:]; bins=25, title="Timepoint $t",ylabel="Frequency", legend=:none, xlims=opts.encoding.range)
            else
                h = histogram(mean(X_norm; dims=1)[:]; bins=25, title="Average of All Observations",ylabel="Frequency", legend=:none, xlims=opts.encoding.range)
            end

            p = plot(h, p; layout=(2,1))

        end
    
        push!(ps, p)
    end

    summary = plot(ps..., layout=(1, length(tis)), size=size, bottom_margin=(padding)mm, left_margin=(padding)mm, top_margin=(padding)mm, kwargs...)

    if return_subplots
        pushfirst!(ps, summary)
        return basis_per_time, ps
    else
        return basis_per_time, [summary]
    end
    


    return basis_per_time, summary
end

"""
```Julia
plot_encoding(E::Union(Symbol, MPSTime.Encoding), 
              d::Integer, 
              X_train::Matrix{Float64}=zeros(0,0), 
              y_train::Vector{Any}=[];
              <keyword arguments>) -> encoding::Vector, plots::Vector{Plots.plot}
```

Plot the first `d` terms of the encoding `E` across its entire domain, `X_train` and `y_train` are only needed if `E` is data driven or time dependent, or `plot_hist` is true.

# Return Values
- `encoding::Vector`: The value of the encoding computed at each time point.
- `plots::Vector{Plots.plot}` A vector of plots, containing either a summary figure, or a summary figure followed by every subplot if \
`return_subplots=true`.

# Keyword Arguments
- `plot_hist::Bool=E.isdatadriven`: Whether to plot the histogram of the traing data at several time points. Useful for understanding the behviour of data-driven bases.
- `tis::Vector{<:Integer} = Int[]`: Time(s) to plot the Encoding at.
- `ds::Vector{<:Integer} = collect(1:d)`: Enables plotting of a subset of a d-dimensional Encoding, e.g. `ds=[1,3,5]` plots the first, third and fifth basis functions.
- `num_xvals::Integer=500`: Number of points to compute each basis function at.
- `size::Tuple=(1200, 800)`: Size of the summaryt figure in points (passed directly to `plot`).
- `padding::Real=6.`: Size of the padding between the plots and the margins in the summary figure (in mm).
## Used for data-driven Encodings
- `sigmoid_transform::Bool=false`: Whether to apply a robust sigmoid transform to the training data, see [`MPSOptions`](@ref).
- `minmax::Bool=true`: Whether to apply a minmax normalsation to the training data after the sigmoid, see [`MPSOptions`](@ref).
- `data_bounds::Tuple{<:Real, <:Real}=(0.,1.)`: Whether to apply a robust sigmoid transform to the X data, see [`MPSOptions`](@ref).
- `project_basis::Bool=false`: Whether to project the basis onto the data. Supported only by :Legendre and :Fourier basis when `E` has type Symbol.
- `aux_basis_dim::Integer=2`: Dimension of each auxilliary basis. Only relevent when `E` is a `MPSTime.SplitBasis`. 
- `return_subplots::Bool=false`: Wheter to return all of the subplots alongside the summary figure.W

- `kwargs`: Passed to Plots.Plot()

"""
plot_encoding(s::Symbol, args...; project_basis::Bool=false, kwargs...) = plot_encoding(model_encoding(s, project_basis), args...; kwargs...)
