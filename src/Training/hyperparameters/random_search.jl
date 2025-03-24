function make_grid(
    rng::AbstractRNG,
    grid_type::Symbol, 
    lb::AbstractVector{N}, 
    ub::AbstractVector{N}, 
    is_disc::AbstractVector{Bool},
    types::AbstractVector{Type}, 
    maxiters::Integer;
    maxrerolls::Integer=100
    ) where N <: Number


    if grid_type == :UniformRandom
        # check that number of samples is less than exhuastive search
        samps = Vector{<:Vector{N}}(undef, maxiters)
        for i in 1:maxiters
            nrolls = 1
            success = false
            samp = Vector{N}(undef, length(lb))

            while ~success && nrolls <= maxrerolls
                for j in eachindex(samp)
                    if is_disc[j]
                        samp[j] = sample(rng, lb[j]:ub[j])
                    else 
                        samp[j] = (ub[j] - lb[j]) * rand(rng) + lb[j]
                    end
                end
                if samp in samps[1:i-1]
                    nrolls += 1
                else
                    success = true
                end
            end
            if ~success
                @warn "Skipped sample $i/$maxiters as it wasn't unique after $maxrerolls attempts"
            else
                samps[i] = samp
            end
        end
        
        return samps

    elseif grid_type == :LatinHypercube
        dims = Vector{LHS.LHCDimension}(undef, length(lb))
        for i in eachindex(lb)
            if is_disc[i]
                dims[i] = LHS.Categorical(round(Int, ub[i] - lb[i] + 1))
            else
                dims[i] = LHS.Continuous()
            end
        end
        LHC = LHS.randomLHC(rng, maxiters, dims)

        LHCs = LHS.scaleLHC(LHC, map(tuple, lb, ub))

        return collect(eachrow(LHCs))

    elseif grid_type == :Exhaustive
        if all(is_disc)
            return reduce(vcat, Iterators.product(map(range, lb, ub)...))
        else
            throw(ArgumentError("All hyperparameters must be discrete if using the :Exhaustive search method"))
        end

    else
        throw(ArgumentError("Unknown sampling type, expected :LatinHypercube, :UniformRandom, or :Exhaustive")
)
    end
end


function grid_search(
    rng::AbstractRNG,
    objective::Function, 
    method::MPSRandomSearch, 
    lb::AbstractVector{<:Number}, 
    ub::AbstractVector{<:Number}, 
    is_disc::AbstractVector{Bool},
    types::AbstractVector{Type}, 
    maxiters::Integer,
    distribute_iters::Bool=true;
    maxrerolls::Integer=100,
    )

    trials = make_grid(rng, method.sampling, lb, ub, is_disc, types, maxiters; maxrerolls=maxrerolls)

    losses = Vector{Float64}(undef, length(trials))
    min_ind = -1
    min_loss = Inf64
    
    if distribute_iters
        losses .= pmap(objective, trials)
        min_ind = argmin(losses)
    else
        for i in eachindex(trials)
            losses[i] = objective(trials[i])
            if losses[i] < min_loss
                min_loss = losses[i]
                min_ind = i
            end
        end
    end

    return trials[min_ind]
end