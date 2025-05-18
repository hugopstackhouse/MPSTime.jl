# define equality and approximate equality for custom types
# this was done with metaprogramming previously but that causes nothing but trouble

function ==(e1::EncodedTimeSeriesSet, e2::EncodedTimeSeriesSet)
    return e1.class_distribution == e2.class_distribution && ==(e1.original_data, e2.original_data) && ==(e1.timeseries, e2.timeseries)
end


function ==(p1::PState, p2::PState) 
    return p1.label == p2.label && p1.label_index == p2.label_index && ==(p1.pstate, p2.pstate)
end

function ==(m1::TrainedMPS, m2::TrainedMPS)

    return m1.opts == m2.opts && ==(m1.train_data, m2.train_data) && ==(m1.mps.data, m2.mps.data)
end
        
    
# isapprox is only used for numeric structs that will serve as a source of floating point error, the uses of "==" in its definition are intentional
function isapprox(e1::EncodedTimeSeriesSet, e2::EncodedTimeSeriesSet)
    return e1.class_distribution == e2.class_distribution && isapprox(e1.original_data, e2.original_data) && isapprox(e1.timeseries, e2.timeseries)
end

# I could not tell you why generic elementwise isapprox seems to be broken by LinearAlgebra.jl, but here we are
function isapprox(it1::TimeSeriesIterable, it2::TimeSeriesIterable)
    return length(it1) == length(it2) && all(map(isapprox, it1, it2))
end

function isapprox(p1::PState, p2::PState) 
    return p1.label == p2.label && p1.label_index == p2.label_index && isapprox(p1.pstate, p2.pstate)
end

function isapprox(m1::TrainedMPS, m2::TrainedMPS)

    return m1.opts == m2.opts && isapprox(m1.train_data, m2.train_data) && isapprox(m1.mps.data, m2.mps.data)
end
