module MPSTime

# Import these two libraries First and in this order
using GenericLinearAlgebra
# using MKL (Banished for crimes)

using LinearAlgebra # low level blas algorithms
using Strided # strided array support
using ITensors 
using NDTensors # Library that ITensors is built on, used for some hacks
using Distributed # hyperparameter tuning reasons
using StaticArrays # faster imputation
using LoopVectorization # fast custom linear algebra functions

using Random
using StableRNGs # Used for some hyperparameter tuning/MLJ


# Allow Optimisation algorithms from external libraries
#TODO replace with the Optimization.jl versions
using Optim
using OptimKit
using Normalization # Standardised normalisation by Brendan :). Used to do the preprocessing / denormalising steps

# Libraries for hyperparameter tuning
using Optimization
using OptimizationOptimJL
import LatinHypercubeSampling as LHS # for random search
import MLJBase


using LegendrePolynomials # For legendre polynomial basis
using KernelDensity # Used for histogram-derived data-driven bases


#TODO remove some of these
using QuadGK # used to be used in imputation.jl, may be redundant
using NumericalIntegration # Used extensively in imputation.jl
using Integrals # Not sure if used (remove)
import SciMLBase # used in encodings for SL coeffs, (remove)

import Base.convert # allows converting BBOpt to string in structs.jl, and converting the abstract MPSOptions to a concrete Options type
import Base.== # for comparing custom datatypes, used for defining equality as 'stored data is equal in numeric value' for testing reasons. This is done in Structs/operations.jl
import Base.isapprox # same as above, only fuzzy equality to avoid FPerror problems
import Base.eltype # for determining t

import MLBase # For some useful multivariate stats calculations
using StatsBase: countmap, sample, median, pweights
using Distributions # Used to compute stats in stats.jl and the imputation library


using JLD2 # for save/load
using StatsPlots, Plots, Plots.PlotMeasures # Plotting in imputation.jl
using LaTeXStrings # Equation formatting in plot titles/axes
using PrettyTables # Nicely formatted training / imputation text output 
import ProgressMeter 
using Markdown # Documentation

using Tables # Used for MLJ
using MLJ # Used for MLJ Integration
import MLJModelInterface # MLJ Integration
using Documenter # pretty docstrings

# Custom Data Structures and types - include first
include("Structs/structs.jl") # Structs used to hold data during training, useful value types, and wrapper types like "BBOpt".
include("Encodings/basis_structs.jl") # Definition of "Encoding", "Basis", etc
include("Structs/options.jl") # Options and MPSOptions types, require the "Encoding" type to be defined. Also defines "TrainedMPS", which requires "MPSOptions" to be defined already.
include("Structs/operations.jl") # includes definitions of "==" and "isapprox" for the custom datatypes.

# Structs for Legacy ITensor
include("legacy_itensor/Structs/structs.jl") # Old structs defined with ITensor 
include("legacy_itensor/Structs/operations.jl") # Operations like '==' and 'approx' on old ITensor based structs

# Functions and structs used to define basis functions / encodings
include("Encodings/encodings.jl")
include("Encodings/bases.jl")
include("Encodings/splitbases.jl")


include("summary.jl") # Some useful stats functions
include("utils.jl") # Some utils used by the entire library

# Visualisation utilities
include("Vis/vis_encodings.jl")

# Analysis utilities
include("Analysis/analyse.jl")

include("Training/loss_functions.jl") # Where loss functions and the LossFunction type are defined
include("Training/RealRealHighDimension.jl") # The training algorithm, fitMPS and co

# Legacy ITensor Implementation
include("legacy_itensor/loss_functions.jl") # Loss functions for ITensor based encoding
include("legacy_itensor/summary.jl") # 
include("legacy_itensor/RealRealLegacyITensor.jl") # Legacy fitmps


# Imputation
include("Imputation/imputation.jl") # Some structs, and scaffolds for setting up and solving ImpuationProblems
include("Imputation/metrics.jl"); # Metrics used to measure imputation performance
include("Imputation/MPS_methods.jl"); # contains the functions to project states onto an MPS / get most likely states for a region
include("Imputation/sampling_utils.jl"); # contains functions to compute a rdm matrix from an MPS, and pretty much every way you can think of to sample from it
include("Imputation/hyperparameter_opt.jl");

# Simulation
include("Simulation/missing_data_mechanisms.jl"); # contains functions to simulate various mechansims of missing data.
include("Simulation/toy_data.jl"); # functions to simulate synthetic data


# hyperparameter tuning
include("Training/hyperparameters/hyperopt_utils.jl")
include("Training/hyperparameters/random_search.jl")
include("Training/hyperparameters/tuning.jl")
include("Training/hyperparameters/evaluate.jl")

# MLJ
include("MLJIntegration/MLJ_integration.jl") # MLJ Integration
include("MLJIntegration/MLJ_utils.jl")
# include("MLJIntegration/imputation_hyperopt_hack.jl") # Hyperoptimising imputation using MAE. MLJ was not designed for this at all 


export 
    # Structs
    MPSOptions,
    TrainedMPS,
    EncodedTimeSeriesSet,
    Encoding, # so help(Encoding) gives useful information

    # functions to construct Bases
    stoudenmire,
    fourier,
    legendre,
    legendre_no_norm,
    sahand,
    uniform,
    function_basis,
    histogram_split,
    uniform_split,

    # nicley formatted training summaries
    get_training_summary,
    sweep_summary,
    print_opts,

    classify, # classify unseen data

    # vis
    plot_encoding,

    # analysis
    bipartite_spectrum,
    single_site_spectrum,
    see_variation,

    # Training functions
    fitMPS, # gotta fit those MPSs somehow

    # Imputation
    init_imputation_problem, # generate and collect data necessary for imputation
    MPS_impute, # The main imputation function, all functionality can be accessed hyperparameter
    get_cdfs, # compute the reduced density matrix (as a cdf) at every site. Useful for data vis/debugging weird imputations
    ImputationProblem,

    # Simulation
    mcar, # simulate missing completely at random mechanism for imputation
    mnar, # simulate missing not at random mechanism for imputation
    mar, # simulate missing at random mechanism for imputation
    trendy_sine, # simulate noise corrupted trendy sinusoid
    state_space, # Generate time series from a state space model with residual terms drawn from a normal distribution

    # hyperparameter tuning
    tune,
    evaluate,
    is_omp_threading,
    eval_loss,
    ImputationLoss,
    MisclassificationRate,
    BalancedMisclassificationRate,
    MPSRandomSearch,
    # MLJ 
    MPSClassifier
end
