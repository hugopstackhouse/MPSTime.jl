# encoding_names = [:uniform, :stoudenmire, :legendre, :fourier, :legendre_norm, histogram_split(:fourier), uniform_split(:legendre), :sahand_legendre, :SLTD]
"""
    Encoding

Abstract supertype of all encodings. To specify an encoding for MPS training, set the `encoding` keyword when calling `MPSOptions`.
# Example
```
julia> opts = MPSOptions(; encoding=:Legendre);
julia> W, info, test_states = fitMPS( X_train, y_train, X_test, y_test, opts);
```
# Encodings
- `:Legendre`: The first *d* L2-normalised [Legendre Polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials). Real valued, and supports passing `projected_basis=true` to `MPSOptions`.
- `:Fourier`: Complex valued Fourier coefficients. Supports passing `projected_basis=true` to `MPSOptions`.


```math
    \\Phi(x; d) = \\left[1. + 0i, e^{i \\pi x}, e^{-i \\pi x}, e^{2i \\pi x}, e^{-2i \\pi x}, \\ldots \\right] / \\sqrt{d} 
```

- `:Stoudenmire`: The original complex valued "Spin-1/2" encoding from Stoudenmire & Schwab, 2017 [arXiv](http://arxiv.org/abs/1605.05775). Only supports *d* = 2

```math
    \\Phi(x) = \\left[ e^{3 i \\pi x / 2} \\cos(\\frac{\\pi}{2} x),  e^{-3 i \\pi x / 2} \\sin(\\frac{\\pi}{2} x)\\right]
```

- `:Sahand_Legendre_Time_Dependent`:  (:SLTD) A custom, real-valued encoding constructed as a data-driven adaptation of the Legendre Polynomials. At each time point, ``t``, the training data is used to construct a probability density function that describes the distribution of the time-series amplitude ``x_t``. This is the first basis function. 

    ``b_1(x; t) = \\text{pdf}_{x_t}(x_t)``. This is computed with KernelDensity.jl:
```
julia> Using KernelDensity
julia> xs_samps = range(-1,1, max(200,size(X_train,2)))
julia> b1(xs,t) = pdf(kde(X_train[t,:]), xs_samps)
```
The second basis function is the first order polynomial that is L2-orthogonal to this pdf on the interval [-1,1]. 

```math
b_2(x;t) = a_1 x + a_0 \\text{ where } \\int_{-1}^1 b_1(x;t) b_2^*(x; t) \\textrm{d} x = 0, \\ \\lvert\\lvert b_2(x; t) \\rvert\\rvert_{L2} = 1
```

The third basis function is the second order polynomial that is L2-orthogonal to the first two basis functions on [-1,1], etc.

-`:Custom`: For use with user-defined custom bases. See [`function_basis`](@ref)
"""
abstract type Encoding end

struct Basis <: Encoding 
    name::String
    init::Function
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isdatadriven::Bool
    range::Tuple{Real, Real}
end


function Base.show(io::IO, E::Basis)
    print(io,"Basis($(E.name))")
end

# Splitting up into a time dependent histogram
struct SplitBasis <: Encoding
    name::String
    init::Union{Function}
    splitmethod::Function
    aux_enc::Encoding
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isdatadriven::Bool
    range::Tuple{Real, Real}
    SplitBasis(s::String, init::Function, spm::Function, aux_enc::Encoding, encode_func::Function, isc::Bool, istd::Bool, isdd::Bool, range::Tuple{<:Real, <:Real}) = begin

        if aux_enc.iscomplex != isc
            error("The SplitBasis and its auxilliary basis must agree on whether they are complex!")
        end

        if aux_enc.range != range #TODO This is probably not actually necessary, likely could be handled in encode_TS?
            error("The SplitBasis and its auxilliary basis must agree on the normalised timeseries range!")
        end
        if aux_enc.isdatadriven || aux_enc.istimedependent
            error("Splitting up a data-driven encoding is not yet supported, sorry")
        end

        isdd |= aux_enc.isdatadriven

        new(s, init, spm, aux_enc, encode_func, isc, istd, isdd, range)
    end
end

function Base.show(io::IO, E::SplitBasis)
    print(io,"SplitBasis($(E.name))")
end


##############################################

function stoudenmire()
    sl = "Stoudenmire" 
    enc = angle_encode
    iscomplex=true
    istimedependent=false
    isdatadriven = false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end


function fourier(; project=false)
    sl = project ? "Projected Fourier" : "Fourier"
    enc = fourier_encode
    iscomplex=true
    istimedependent=project
    isdatadriven = project
    range = (-1,1)
    init = project ? project_fourier : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function legendre(; norm=false, project=false)
    sl = norm ? "Legendre_Norm" : "Legendre"
    sl = project ? "Projected "* sl : sl
    enc = norm ? legendre_encode : legendre_encode_no_norm
    iscomplex = false
    istimedependent=project
    isdatadriven = project
    range = (-1,1)
    init = project ? project_legendre : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

legendre_no_norm(; project=false) = legendre(; norm=false, project) 

function sahand_legendre(istimedependent::Bool=true)
    sl = "Sahand-Legendre" * (istimedependent ? " Time Dependent" : " Time Independent")
    enc = sahand_legendre_encode
    iscomplex = false
    istimedependent=istimedependent
    isdatadriven=true
    range = (-1,1)
    init = istimedependent ?  init_sahand_legendre_time_dependent : init_sahand_legendre

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function sahand()
    sl = "Sahand"
    enc = sahand_encode
    iscomplex=true
    istimedependent=false
    isdatadriven=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function uniform()
    sl = "Uniform"
    enc = uniform_encode
    iscomplex = false
    istimedependent=false
    isdatadriven=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

# the error function Basis, raises an error (used as a placeholder only)
function erf()
    f = _ -> error("Tried to use a basis that isn't implemented")
    iscomplex = false # POSIX compliant error function
    istimedependent = false
    isdatadriven = false
    range = (-1,1)
    return Basis("Pun Intended", no_init, f, iscomplex, istimedependent, isdatadriven, range)
end


"""

    function_basis(basis::Function, is_complex::Bool, range::Tuple{<:Real,<:Real}, <args>; name::String="Custom")

Constructs a time-(in)dependent encoding from the function `basis`, which is either real or complex, and has support on the interval `range`.

For a time independent basis, the input function must have the signature :

    basis(x::Float64, d::Integer, init_args...) 
    
and return a d-dimensional numerical Vector. 
A vector ``[x_1, x_2, x_3, ..., x_N]`` will be encoded as ``[b(x_1), b(x_2), b(x_3),..., b(x_N)]``

To use a time dependent basis, set `is_time_dependent` to true. The input function must have the signature 

    basis(x::Float64, d::Integer, ti::Int, init_args...) 

and return a d-dimensional numerical Vector.
A vector ``[x_1, x_2, x_3, ..., x_N]`` will be encoded as  ``[b_1(x_1), b_2(x_2), b_3(x_3),..., b_N(x_N)]``

# Optional Arguments
- `is_time_dependent::Bool=false`: Whether the basis is time dependent 
- `is_data_driven::Bool=false`: Whether functional form of the basis depends on the training data
- `init::Function=no_init`: The initialiser function for the basis. This is used to compute arguments for the function that are not known in advance,
for example, the polynomial coefficients for the Sahand-Legendre basis. This function should have the form

    init_args = opts.encoding.init(X_normalised::AbstractMatrix, y::AbstractVector; opts::MPSTime.Options=opts)

`X_normalised` will be preprocessed (with sigmoid transform and MinMax transform pre-applied), **with Time series as columns**

# Example
The Legendre Polynomial Basis:

```jldoctest
using MPSTime, LegendrePolynomials
function legendre_encode(x::Float64, d::Int)
    # default legendre encoding: choose the first n-1 legendre polynomials

    leg_basis = [Pl(x, i; norm = Val(:normalized)) for i in 0:(d-1)] 
    
    return leg_basis
end
custom_basis = function_basis(legendre_encode, false, (-1., 1.))
# output
Basis(Custom)
```

"""
function function_basis(
    basis::Function, 
    is_complex::Bool, 
    range::Tuple{<:Real,<:Real}, 
    is_time_dependent::Bool=false, 
    is_data_driven::Bool=false,
    init::Function=no_init; 
    name::String="Custom" )
    return Basis(name, init, basis, is_complex, is_time_dependent, is_data_driven, range)
end


function histogram_split(aux_enc::Encoding)
    isc = aux_enc.iscomplex
    range = aux_enc.range

    name = "Hist Split $(aux_enc.name)" 

    init = split_init
    splitmethod = hist_split
    istd = true
    isdatadriven = true
    encode_func = project_onto_bins

    return SplitBasis(name, init, splitmethod, aux_enc, encode_func, isc, istd, isdatadriven, range)
end

function uniform_split(aux_enc::Encoding)
    isc = aux_enc.iscomplex
    range = aux_enc.range

    name = "Unif Split $(aux_enc.name)" 

    init = split_init
    splitmethod = unif_split
    istd = aux_enc.istimedependent # if the aux.enc _is_ time dependent we have to treat the entire split as such
    isdatadriven = aux_enc.isdatadriven
    encode_func = project_onto_bins

    return SplitBasis(name, init, splitmethod, aux_enc, encode_func, isc, istd, isdatadriven, range)

end

histogram_split(s::Symbol) = symbolic_encoding(histogram_split(model_encoding(s)))
uniform_split(s::Symbol) = symbolic_encoding(uniform_split(model_encoding(s)))


histogram_split() = histogram_split(uniform())
uniform_split() = uniform_split(uniform())
