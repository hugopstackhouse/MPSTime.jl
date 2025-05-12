using JLD2
using Statistics
@load "Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test


opts = MPSOptions(verbosity=-1, log_level=0)
mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts)
mps2, _... = fitMPS(X_train, y_train, opts)

c1 = classify(mps, test_states)
c2 = classify(mps, X_test)
c3 = classify(mps2, X_test)

perm = sortperm(y_test)


@test c1 == c2[perm]
@test c2 == c3

@test isapprox(mean(c2 .== y_test), 0.9582118561710399)

# conf = c0_correct, c0_incorrect, c1_correct, c1_incorrect
conf = zeros(Int, 4)
for i in eachindex(y_test)
    c_true = y_test[i]
    c_pred = c2[i]
    if c_true == 0
        if c_pred == 0
            conf[1] += 1
        else
            conf[2] += 1
        end

    elseif c_pred == 1
        conf[3] += 1
    else
        conf[4] += 1
    end
end

@test conf == Int[496, 17, 490, 26]