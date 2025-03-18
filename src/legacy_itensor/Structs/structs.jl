# ITensor type aliases
const PCacheIT = Matrix{ITensor}
const PCacheColIT = AbstractVector{ITensor} # for view mapping shenanigans

# data structures
struct PStateIT
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Any # TODO make this a fancy scientific type
    label_index::UInt
end

function PStateIT(ps::PState, site_indices::AbstractVector{Index{Int64}})
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate = ps.pstate
    its = [ITensor(pstate[i], site_indices[i]) for i in eachindex(pstate)]
    return PStateIT(MPS(its), ps.label, ps.label_index)
end

function PState(ps::PStateIT)
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate = ps.pstate
    vs = [vector(s) for s in pstate]
    return PState(vs, ps.label, ps.label_index)
end


const TimeSeriesIterableIT = Vector{PStateIT}

function to_TimeSeriesIterableIT(ts::TimeSeriesIterable, site_indices::AbstractVector{Index{Int64}})
    return [PStateIT(t, site_indices) for t in ts]
end

function to_TimeSeriesIterable(ts::TimeSeriesIterableIT)
    return PState.(ts)
end

"""
    EncodedTimeSeriesSet

Holds an encoded time-series dataset, as well as a copy of the original data and its class distribution.
"""
struct EncodedTimeSeriesSetIT
    timeseries::TimeSeriesIterableIT
    original_data::Matrix{Float64}
    class_distribution::Vector{<:Integer}
end

function EncodedTimeSeriesSetIT(class_dtype::DataType=Int64) # empty version
    tsi = TimeSeriesIterableIT(undef, 0)
    mtx = zeros(0,0)
    class_dist = Vector{class_dtype}(undef, 0) # pays to assume the worst and match types...
    return EncodedTimeSeriesSetIT(tsi, mtx, class_dist)
end

Base.isempty(e::EncodedTimeSeriesSetIT) = isempty(e.timeseries) && isempty(e.original_data) && isempty(e.class_distribution)

function EncodedTimeSeriesSetIT(ETS::EncodedTimeSeriesSet, site_indices::AbstractVector{Index{Int64}})
    tsi = to_TimeSeriesIterableIT(ETS.timeseries, site_indices)

    return EncodedTimeSeriesSetIT(tsi, ETS.original_data, ETS.class_distribution)
end

function EncodedTimeSeriesSet(ETS::EncodedTimeSeriesSetIT)
    tsi = to_TimeSeriesIterable(ETS.timeseries)

    return EncodedTimeSeriesSet(tsi, ETS.original_data, ETS.class_distribution)
end

