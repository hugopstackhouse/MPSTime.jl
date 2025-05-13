# define equality and approximate equality for custom types
# this was done with metaprogramming previously but that causes nothing but trouble

function ==(e1::EncodedTimeSeriesSetIT, e2::EncodedTimeSeriesSetIT)
    return e1.class_distribution == e2.class_distribution && ==(e1.original_data, e2.original_data) && ==(e1.timeseries, e2.timeseries)
end

function ==(p1::PStateIT, p2::PStateIT) 
    return p1.label == p2.label && p1.label_index == p2.label_index && ==(p1.pstate.data, p2.pstate.data)
end

        

# I could not tell you why generic elementwise isapprox seems to be broken by LinearAlgebra.jl, but here we are
function isapprox(it1::TimeSeriesIterableIT, it2::TimeSeriesIterableIT)
    return length(it1) == length(it2) && all(map(isapprox, it1, it2))
end

# isapprox is only used for numeric structs that will serve as a source of floating point error, the uses of "==" in its definition are intentional
function isapprox(e1::EncodedTimeSeriesSetIT, e2::EncodedTimeSeriesSetIT)
    return e1.class_distribution == e2.class_distribution && isapprox(e1.original_data, e2.original_data) && isapprox(e1.timeseries, e2.timeseries)
end

function isapprox(p1::PStateIT, p2::PStateIT) 
    return p1.label == p2.label && p1.label_index == p2.label_index && isapprox(p1.pstate.data, p2.pstate.data)
end