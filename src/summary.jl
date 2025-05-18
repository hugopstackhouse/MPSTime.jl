const AnyPState = Union{PState, PStateIT}
const AnyTimeSeriesIterable = Union{TimeSeriesIterable, TimeSeriesIterableIT}

function contract_mps(W::MPS, PS::PState)
        N_sites = length(W)
        s = get_siteinds(W)
        res = ITensor(1.)
        for i=1:N_sites
            res *= W[i] * itensor(conj(PS.pstate[i]), s[i])
        end

        return res 

end

function contract_mps(W::MPS, PS::PState, pos::Integer, label_onehot::ITensor)
    N_sites = length(W)
    s = get_siteinds(W)
    res = ITensor(1.)
    for i=1:N_sites
        if i == pos
            res *= W[i] * label_onehot * itensor(conj(PS.pstate[i]), s[i])
        else
            res *= W[i] * itensor(conj(PS.pstate[i]), s[i])
        end
    end

    return res[1]

end


function MSE_loss_acc_iter(W::MPS, PS::AnyPState, label_idx::Index)
    """For a given sample, compute the Quadratic Cost and whether or not
    the corresponding prediction (using argmax on deicision func. output) is
    correctly classfied"""
    shifted_label = PS.label_index # ground truth label
    y = onehot(label_idx => shifted_label)


    yhat = contract_mps(W, PS)
    
    diff_sq = abs2.(vector(yhat - y))
    sum_of_sq_diff = real(sum(diff_sq))

    mse_loss = 0.5 * sum_of_sq_diff
    kld_loss = -log(abs2(yhat[label_idx=>shifted_label]))

    # now get the predicted label
    correct = 0
    
    if argmax(abs.(vector(yhat))) == shifted_label
        correct = 1
    end

    return [mse_loss, kld_loss, correct]

end

function MSE_loss_acc(W::MPS, PSs::AnyTimeSeriesIterable)
    """Compute the MSE loss and accuracy for an entire dataset"""
    pos, label_idx = find_label(W)
    mse_loss, kld_loss, acc = reduce(+, MSE_loss_acc_iter(W, PS, label_idx) for PS in PSs)
    mse_loss /= length(PSs)
    kld_loss/= length(PSs)
    acc /= length(PSs)

    return mse_loss, kld_loss, acc 

end

function MSE_loss_acc_conf_iter!(W::MPS, PS::AnyPState, label_idx::Index, conf::Matrix)
    """For a given sample, compute the Quadratic Cost and whether or not
    the corresponding prediction (using argmax on decision func. output) is
    correctly classfified"""
    shifted_label = PS.label_index # ground truth label
    y = onehot(label_idx => shifted_label)


    yhat = contract_mps(W, PS)
    
    diff_sq = abs2.(array(yhat - y))
    sum_of_sq_diff = real(sum(diff_sq))

    mse_loss = 0.5 * sum_of_sq_diff
    kld_loss = -log(abs2(yhat[label_idx=>shifted_label]))


    # now get the predicted label
    correct = 0
    pred = argmax(abs.(vector(yhat)))
    if pred == shifted_label
        correct = 1
    end

    conf[shifted_label, pred] += 1

    return [mse_loss, kld_loss, correct]

end

function MSE_loss_acc_conf(W::MPS, PSs::AnyTimeSeriesIterable)
    pos, label_idx = find_label(W)
    nc = ITensors.dim(label_idx)
    conf = zeros(Int, nc,nc)

    mse_loss, kld_loss, acc = reduce(+, MSE_loss_acc_conf_iter!(W, PS, label_idx, conf) for PS in PSs)
    mse_loss /= length(PSs)
    kld_loss /= length(PSs)
    acc /= length(PSs)

    return mse_loss, kld_loss, acc, conf

end

function classify(mps::TrainedMPS, test_states::EncodedTimeSeriesSet)

    pss = test_states.timeseries
    # Ws, l_ind = expand_label_index(mps.mps)

    pss_train = mps.train_data.timeseries

    labels = sort(unique([ps.label for ps in pss_train]))


    preds = Vector{Int64}(undef, length(pss))
    for (i, ps) in enumerate(pss)
        yhat = contract_mps(mps.mps, ps)
        pred = argmax(abs2.(vector(yhat)))
        preds[i] = labels[pred]

    end

    return preds
        
end

"""
```Julia
classify(mps::TrainedMPS, X_test::AbstractMatrix)) -> (predictions::Vector)
```
Use the `mps` to predict the class of the rows of `X_test` by computing the maximum overlap.

# Example
```julia-repl
julia> W, info, test_states = fitMPS( X_train, y_train);

julia> preds  = classify(W, X_test); # make some predictions

julia> mean(preds .== y_test)
0.9504373177842566
```

"""
function classify(mps::TrainedMPS, X_test::AbstractMatrix)
    opts = safe_options(mps.opts) # make sure options is abstract
    opts = _set_options(opts; verbosity=-10)

    X_train = mps.train_data.original_data
    X_train_scaled, X_test_scaled, norms, oob_rescales = transform_data(permutedims(X_train), permutedims(X_test); opts=opts)

    sites = get_siteinds(mps.mps)
    pss_train = mps.train_data.timeseries

    y_train = [ps.label for ps in pss_train]
    classes = sort(unique(y_train))
    num_classes = length(classes)
    
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes))
    n_tests = size(X_test_scaled,2)

    s = EncodeSeparate{opts.encode_classes_separately}()
    _, enc_args_tr = encode_dataset(s, X_train, X_train_scaled, y_train, "train", sites; opts=opts, class_keys=class_keys)
    test_states, _ = encode_dataset(s, X_test, X_test_scaled, fill(-1, n_tests), "test", sites; opts=opts, class_keys=Dict(-1=> n_tests), training_encoding_args=enc_args_tr)
    return classify(mps, test_states)
end

@deprecate classify(mps::TrainedMPS, X_test::AbstractMatrix, opts::AbstractMPSOptions) classify(mps, X_test)


function classify_overlap(Ws::Vector{MPS}, pss::AnyTimeSeriesIterable)
    # mps0 overlaps with ORIGINAL class 0 and mps1 overlaps with ORIGINAL class 1
    # preds are in terms of label_index not label!
    @assert all(length(Ws[1]) .== length.(Ws)) "MPS lengths do not match!"

    preds = Vector{Int64}(undef, length(pss))
    all_overlaps = Vector{Vector{Float64}}(undef, length(pss))
    for i in eachindex(pss)
        overlaps = Vector{Float64}(undef, length(Ws))
        for (wi,w) in enumerate(Ws)
            overlaps[wi] = abs(contract_mps(w, pss[i])[1])
        end

        preds[i] = argmax(overlaps)
        all_overlaps[i] = overlaps
    end

    # return overlaps as well for inspection
    return preds, all_overlaps
        
end

function plot_conf_mat(confmat::Matrix)
    reversed_confmat = reverse(confmat, dims=1)
    hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted $n" for n in 0:(size(confmat,2) - 1)]),
        yticks=(1:size(confmat,1), reverse(["Actual n" for n in 0:(size(confmat,1) - 1)]) ),
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Confusion Matrix")
        
    for (i, row) in enumerate(eachrow(reversed_confmat))
        for (j, value) in enumerate(row)
            
            annotate!(j, i, text(string(value), :center, 10))
        end
    end

    display(hmap)
end


function get_training_summary(
    io::Union{IO, Nothing},
    mps::MPS, 
    training_pss::AnyTimeSeriesIterable, 
    testing_pss::AnyTimeSeriesIterable; 
    print_stats=false,
    )
    # get final traing acc, final training loss

    Ws, l_ind = expand_label_index(mps)
    nclasses = length(Ws)

    preds_training, overlaps = classify_overlap(Ws, training_pss)
    true_training = [x.label_index for x in training_pss] # get ground truths
    acc_training = sum(true_training .== preds_training)/length(training_pss)
    
    labels = sort(unique([x.label for x in training_pss]))

    # get final testing acc
    preds_testing, overlaps = classify_overlap(Ws, testing_pss)
    true_testing = [x.label_index for x in testing_pss] # get ground truths

    # get overlap between mps classes
    overlapmat = Matrix{Float64}(undef, nclasses, nclasses)
    for i in eachindex(Ws), j in eachindex(Ws)
        overlapmat[i,j] = abs(dot(Ws[i], Ws[j])) # ITensor dot product conjugates the first argument
    end

    
    confmat = MLBase.confusmat(nclasses, (true_testing), (preds_testing )) 


    # NOTE CONFMAT IS R(i, j) == countnz((gt .== i) & (pred .== j)). So rows (i) are groudn truth and columns (j) are preds
    # tables 
    if isnothing(io) # As I rather unpleasantly discovered in testing: pretty_table(stdin,foo) =/= pretty_table(foo)
        println()
        pretty_table(
            overlapmat;
            title="Overlap Matrix",
            title_alignment=:c,
            title_same_width_as_table=true,
            header = ["|ψ$n⟩" for n in labels],
            row_labels = ["⟨ψ$n|" for n in labels],
            alignment=:c,
            body_hlines=Vector(1:nclasses),
            highlighters = Highlighter(f      = (data, i, j) -> (i == j),
            crayon = crayon"bold" ),
            formatters = ft_printf("%5.3e")
        )

        pretty_table(
            confmat;
            title="Confusion Matrix",
            title_alignment=:c,
            title_same_width_as_table=true,
            header = ["Pred. |$n⟩" for n in labels],
            row_labels = ["True |$n⟩" for n in labels],
            body_hlines=Vector(1:nclasses),
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green" )
        )

    else
        println(io)
        pretty_table(
            io,
            overlapmat;
            title="Overlap Matrix",
            title_alignment=:c,
            title_same_width_as_table=true,
            header = ["|ψ$n⟩" for n in labels],
            row_labels = ["⟨ψ$n|" for n in labels],
            alignment=:c,
            body_hlines=Vector(1:nclasses),
            highlighters = Highlighter(f      = (data, i, j) -> (i == j),
            crayon = crayon"bold" ),
            formatters = ft_printf("%5.3e")
        )

        pretty_table(io,
            confmat;
            title="Confusion Matrix",
            title_alignment=:c,
            title_same_width_as_table=true,
            header = ["Pred. |$n⟩" for n in labels],
            row_labels = ["True |$n⟩" for n in labels],
            body_hlines=Vector(1:nclasses),
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green" )
        )
    end


    # TP, TN, FP, FN FOR TEST SET 
    acc_testing = sum(true_testing .== preds_testing)/length(testing_pss)
    prec = multiclass_precision(preds_testing, true_testing)
    rec = multiclass_recall(preds_testing, true_testing)
    f1 = multiclass_f1score(preds_testing, true_testing)
    specificity = multiclass_specificity(preds_testing, true_testing)
    sensitivity = multiclass_sensitivity(preds_testing, true_testing)
    acc_balanced_testing = balanced_accuracy(preds_testing, true_testing) #

    stats = Dict(
        :train_acc => acc_training,
        :test_acc => acc_testing,
        :test_balanced_acc => acc_balanced_testing,
        :precision => prec,
        :recall => rec,
        :specificity => specificity,
        :f1_score => f1,
        :confmat => confmat
    )

    if print_stats
        statsp = Dict(String(key)=>[stats[key]] for key in filter((s)->s!=:confmat,keys(stats)))
        if isnothing(io)
            pretty_table(statsp)
        else
            pretty_table(io,statsp)
        end
        # println("Testing Accuracy: $acc_testing")
        # println("Training Accuracy: $acc_training")
        # println("Precision: $prec")
        # println("Recall: $rec")
        # println("F1 Score: $f1")
        # println("Specificity: $specificity")
        # println("Sensitivity: $sensitivity")
        # println("Balanced Accuracy: $acc_balanced_testing")
    end
    
    return stats

end

"""
```Julia
get_training_summary(
    [io::IO],
    mps::TrainedMPS, 
    test_states::EncodedTimeSeriesSet;  
    print_stats::Bool=false
    ) -> stats::Dict
```

Print a summary of the training process of `mps`, with performane evaluated on `test_states`.
"""
get_training_summary(io::IO, mps::TrainedMPS, X_test::EncodedTimeSeriesSet;  print_stats::Bool=false) = get_training_summary(io, mps.mps, mps.train_data.timeseries, X_test.timeseries; print_stats=print_stats)
get_training_summary(mps::TrainedMPS, X_test::EncodedTimeSeriesSet;  print_stats::Bool=false) = get_training_summary(nothing, mps.mps, mps.train_data.timeseries, X_test.timeseries; print_stats=print_stats)


"""
```Julia
sweep_summary([io::IO], info)
```
Print a pretty summary of what happened in every sweep

"""
function sweep_summary(io::Union{Nothing,IO}, info)
    
    nsweeps = length(info["time_taken"]) - 2
    row_labels = ["Train Accuracy", "Test Accuracy", "Train KL Div.", "Test KL Div.", "Time taken"]
    header = vcat(["Initial"],["After Sweep $n" for n in 1:(nsweeps)], ["After Norm"], "Mean")

    data = Matrix{Float64}(undef, length(row_labels), nsweeps+3)

    for (i, key) in enumerate(["train_acc", "test_acc", "train_KL_div", "test_KL_div", "time_taken"])
        data[i,:] = vcat(info[key], [mean(info[key][2:end-1])])
    end

    h1 = Highlighter(
        (data, i, j) -> j < length(header) && data[i, j] == maximum(data[i,1:(end-1)]),
        bold       = true,
        foreground = :red 
    )

    h2 = Highlighter(
        (data, i, j) -> j < length(header) && data[i, j] == minimum(data[i,1:(end-1)]),
        bold       = true,
        foreground = :blue
    )
    if isnothing(io)
        pretty_table(
            io,
            data;
            row_label_column_title = "",
            header = header,
            row_labels = row_labels,
            body_hlines = Vector(1:length(row_labels)),
            highlighters = (h1,h2),
            alignment=:c
        )
    else
        pretty_table(
            io,
            data;
            row_label_column_title = "",
            header = header,
            row_labels = row_labels,
            body_hlines = Vector(1:length(row_labels)),
            highlighters = (h1,h2),
            alignment=:c
        )
    end
    #formatters = ft_printf("%.3e"))

end

sweep_summary(info) = sweep_summary(nothing, info)

"""
    print_opts([io::IO], opts::AbstractMPSOptions; long::Bool=false)

Print the MPSOptions struct in a table. Summarises (`long=false`) by default.

"""
function print_opts(opts::AbstractMPSOptions;  long::Bool=false)
    if long
        params = fieldnames(typeof(opts))
    else
        params = [:chi_max, :d, :eta, :nsweeps,:encoding, :sigmoid_transform, :loss_grad]
    end
    optsD = Dict(String(key)=>[getfield(opts, key)] for key in params)
    return pretty_table(optsD)
end

function print_opts(io::IO, opts::AbstractMPSOptions;  long::Bool=false)
    if long
        params = fieldnames(typeof(opts))
    else
        params = [:chi_max, :d, :eta, :nsweeps,:encoding, :sigmoid_transform, :loss_grad]
    end
    optsD = Dict(String(key)=>[getfield(opts, key)] for key in params)
    return pretty_table(io, optsD)
end


function KL_div(W::MPS, test_states::AnyTimeSeriesIterable)
    """Computes KL divergence of TS on MPS"""
    pos, label_index = find_label(W)
    KLdiv = 0.

    for ps in test_states
        qlx = abs2(contract_mps(W, ps, pos, onehot(label_index=>ps.label_index)))
            #qlx = l == 0 ? abs2(dot(x.pstate,W0)) : abs2(dot(x.pstate, W1))
        KLdiv +=  -log(qlx) # plx is 1

    end
    return KLdiv / length(test_states)
end