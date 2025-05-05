
function construct_caches_IT(
        W::MPS, 
        training_pstates::TimeSeriesIterableIT; 
        going_left=true, 
        dtype::DataType=ComplexF64
    )
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the num of training samples to pre-allocate a caching matrix
    N_train = length(training_pstates) 
    # get the number of MPS sites
    N = length(W)

    # pre-allocate left and right environment matrices 
    LE = PCacheIT(undef, N, N_train) 
    RE = PCacheIT(undef, N, N_train)

    if going_left
        # backward direction - initialise the LE with the first site
        for i = 1:N_train
            LE[1,i] =  conj(training_pstates[i].pstate[1]) * W[1] 
        end

        for j = 2 : N
            for i = 1:N_train
                LE[j,i] = LE[j-1, i] * (conj(training_pstates[i].pstate[j]) * W[j])
            end
        end
    
    else
        # going right
        # initialise RE cache with the terminal site and work backwards
        for i = 1:N_train
            RE[N,i] = conj(training_pstates[i].pstate[N]) * W[N]
        end

        for j = (N-1):-1:1
            for i = 1:N_train
                RE[j,i] =  RE[j+1,i] * (W[j] * conj(training_pstates[i].pstate[j]))
            end
        end
    end

    @assert !isa(eltype(eltype(RE)), dtype) || !isa(eltype(eltype(LE)), dtype)  "Caches are not the correct datatype!"

    return LE, RE

end

function update_caches_IT!(
        left_site_new::ITensor, 
        right_site_new::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT, 
        lid::Int, 
        rid::Int, 
        product_states::TimeSeriesIterableIT; 
        going_left::Bool=true
    )
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(product_states)
    num_sites = size(LE)[1]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[num_sites,i] = right_site_new * conj(product_states[i].pstate[rid])
            else
                RE[rid,i] = RE[rid+1,i] * right_site_new * conj(product_states[i].pstate[rid])
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[1,i] = left_site_new * conj(product_states[i].pstate[lid])
            else
                LE[lid,i] = LE[lid-1,i] * conj(product_states[i].pstate[lid]) * left_site_new
            end
        end
    end

end

function decomposeBT_IT(
        BT::ITensor, 
        lid::Int, 
        rid::Int; 
        chi_max=nothing, 
        cutoff=nothing, 
        going_left=true, 
        dtype::DataType=ComplexF64,
        alg::String="divide_and_conquer" # SVD Algorithm to pass to ITensor
    )
    """Decompose an updated bond tensor back into two tensors using SVD"""



    if going_left
        left_site_index = find_index(BT, "n=$lid")
        label_index = find_index(BT, "f(x)")
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            U, S, V = svd(BT, (label_index, left_site_index); maxdim=chi_max, cutoff=cutoff, alg=alg)
        else
            bond_index = find_index(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (bond_index, label_index, left_site_index); maxdim=chi_max, cutoff=cutoff, alg=alg)
        end
        # absorb singular values into the next site to update to preserve canonicalisation
        left_site_new = U * S
        right_site_new = V
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
    else
        # going right, label index automatically moves to the next site
        right_site_index = find_index(BT, "n=$rid")
        label_index = find_index(BT, "f(x)")
        bond_index = find_index(BT, "Link,l=$(lid+1)")


        if isnothing(bond_index)
            V, S, U = svd(BT, (label_index, right_site_index); maxdim=chi_max, cutoff=cutoff, alg=alg)
        else
            V, S, U = svd(BT, (bond_index, label_index, right_site_index); maxdim=chi_max, cutoff=cutoff, alg=alg)
        end
        # absorb into next site to be updated 
        left_site_new = U
        right_site_new = V * S
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
        # @show inds(left_site_new)
        # @show inds(right_site_new)

    end


    return left_site_new, right_site_new

end




function fitMPS_IT(
        W::MPS, 
        training_states_meta::EncodedTimeSeriesSetIT, 
        testing_states_meta::EncodedTimeSeriesSetIT, 
        opts::AbstractMPSOptions=Options(); 
        test_run=false
    ) 
    opts = safe_options(opts) # make sure options is abstract


    verbosity = opts.verbosity
    nsweeps = opts.nsweeps

    if test_run
        verbosity > -1 && println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, []
    end

    # @unpack_Options opts # unpacks the attributes of opts into the local namespace
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    

    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    sites = get_siteinds(W)

    if opts.encode_classes_separately && !opts.train_classes_separately
        @warn "Classes are encoded separately, but not trained separately"
    elseif opts.train_classes_separately && !opts.encode_classes_separately
        @warn "Classes are trained separately, but not encoded separately"
    end

    # check the training states are sorted
    y_train = [ps.label for ps in training_states]
    y_test = [ps.label for ps in testing_states]

    @assert issorted(y_train) "Training data must be sorted by class!"
    @assert issorted(y_test) "Testing data must be sorted by class!"

    has_test = !isempty(y_test)

    verbosity > -1 && println("Using $(opts.update_iters) iterations per update.")
    # construct initial caches
    LE, RE = construct_caches_IT(W, training_states; going_left=true, dtype=opts.dtype)


    # create structures to store training information

    if has_test
        training_information = Dict(
            "train_loss" => Float64[],
            "train_acc" => Float64[],
            "test_loss" => Float64[],
            "test_acc" => Float64[],
            "time_taken" => Float64[], # sweep duration
            "train_KL_div" => Float64[],
            "test_KL_div" => Float64[],
            "test_conf" => Matrix{Float64}[]
        )
    else
        training_information = Dict(
        "train_loss" => Float64[],
        "train_acc" => Float64[],
        "test_loss" => Float64[],
        "time_taken" => Float64[], # sweep duration
        "train_KL_div" => Float64[]
    )
    end

    if opts.log_level > 0

        # compute initial training and validation acc/loss
        init_train_loss, init_KL_div, init_train_acc = MSE_loss_acc(W, training_states)
        
        push!(training_information["train_loss"], init_train_loss)
        push!(training_information["train_acc"], init_train_acc)
        push!(training_information["time_taken"], 0.)
        push!(training_information["train_KL_div"], init_KL_div)


        if has_test 
            init_test_loss, init_test_KL_div, init_test_acc, conf = MSE_loss_acc_conf(W, testing_states)

            push!(training_information["test_loss"], init_test_loss)
            push!(training_information["test_acc"], init_test_acc)
            push!(training_information["test_KL_div"], init_test_KL_div)
            push!(training_information["test_conf"], conf)
        end
    

        #print loss and acc
        if verbosity > -1
            println("Training KL Div. $init_KL_div | Training acc. $init_train_acc.")# | Training MSE: $init_train_loss." )

            if has_test 
                println("Test KL Div. $init_KL_div | Testing acc. $init_test_acc.")#  | Testing MSE: $init_test_loss." )
                println("")
                println("Test conf: $conf.")
            end

        end
    end


    # initialising loss algorithms
    if typeof(opts.loss_grad) <: AbstractArray
        @assert length(opts.loss_grad) == nsweeps "loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps"
        loss_grads = opts.loss_grad
    elseif typeof(opts.loss_grad) <: Function
        loss_grads = [opts.loss_grad for _ in 1:nsweeps]
    else
        error("loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps")
    end

    if opts.train_classes_separately && !(eltype(loss_grads) <: KLDLoss)
        @warn "Classes will be trained separately, but the cost function _may_ depend on measurements of multiple classes. Switch to a KLD style cost function or ensure your custom cost function depends only on one class at a time."
    end

    if typeof(opts.bbopt) <: AbstractArray
        @assert length(opts.bbopt) == nsweeps "bbopt must be an optimiser or an array of optimisers to use with length nsweeps"
        bbopts = opts.bbopt
    elseif typeof(opts.bbopt) <: BBOpt
        bbopts = [opts.bbopt for _ in 1:nsweeps]
    else
        error("bbopt must be an optimiser or an array of optimisers to use with length nsweeps")
    end

    # start the sweep
    update_iters = opts.update_iters
    dtype = opts.dtype
    track_cost = opts.track_cost
    eta = opts.eta
    chi_max = opts.chi_max
    rescale = opts.rescale
    cutoff=opts.cutoff
    for itS = 1:opts.nsweeps
        
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting backward sweeep: [$itS/$nsweeps]")

        for j = (length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[(j+1)] * W[j] # create bond tensor
            # @show inds(BT)
            BT_new = apply_update_IT(
                tsep, 
                BT, 
                LE, 
                RE, 
                j, 
                (j+1), 
                training_states_meta; 
                iters=update_iters, 
                verbosity=verbosity, 
                dtype=dtype, 
                loss_grad=loss_grads[itS], 
                bbopt=bbopts[itS],
                track_cost=track_cost, 
                eta=eta, 
                rescale = rescale
            ) # optimise bond tensor

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT_IT(BT_new, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype, alg=opts.svd_alg)
                
            # update the caches to reflect the new tensors
            update_caches_IT!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
    
        # add time taken for backward sweep.
        verbosity > -1 && println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = construct_caches_IT(W, training_states; going_left=false)
        
        verbosity > -1 && println("Starting forward sweep: [$itS/$nsweeps]")

        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            # @show inds(BT)
            BT_new = apply_update_IT(
                tsep, 
                BT, 
                LE, 
                RE, 
                j, 
                (j+1), 
                training_states_meta; 
                iters=update_iters, 
                verbosity=verbosity, 
                dtype=dtype, 
                loss_grad=loss_grads[itS], 
                bbopt=bbopts[itS],
                track_cost=track_cost, 
                eta=eta, 
                rescale=rescale
            ) # optimise bond tensor

            lsn, rsn = decomposeBT_IT(BT_new, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype, alg=opts.svd_alg)
            update_caches_IT!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        LE, RE = construct_caches_IT(W, training_states; going_left=true)
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS. Time for sweep: $(round(time_elapsed,digits=2))s")

        if opts.log_level > 0

            # compute the loss and acc on both training and validation sets
            train_loss, train_KL_div, train_acc = MSE_loss_acc(W, training_states)


            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)


            if has_test 
                test_loss, test_KL_div, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
        
                push!(training_information["test_loss"], test_loss)
                push!(training_information["test_acc"], test_acc)
                push!(training_information["test_KL_div"], test_KL_div)
                push!(training_information["test_conf"], conf)
            end
        

            if verbosity > -1
                println("Training KL Div. $train_KL_div | Training acc. $train_acc.")#  | Training MSE: $train_loss." )

                if has_test 
                    println("Test KL Div. $test_KL_div | Testing acc. $test_acc.")#  | Testing MSE: $test_loss." )
                    println("")
                    println("Test conf: $conf.")
                end

            end
        end

        if opts.exit_early && train_acc == 1.
            break
        end
       
    end
    normalize!(W)
    verbosity > -1 && println("\nMPS normalised!\n")
    if opts.log_level > 0

        # compute the loss and acc on both training and validation sets
        train_loss, train_KL_div, train_acc = MSE_loss_acc(W, training_states)


        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["time_taken"], NaN)
        push!(training_information["train_KL_div"], train_KL_div)


        if has_test 
            test_loss, test_KL_div, test_acc, conf = MSE_loss_acc_conf(W, testing_states)

            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(training_information["test_conf"], conf)
        end
    

        if verbosity > -1
            println("Training KL Div. $train_KL_div | Training acc. $train_acc.")#  | Training MSE: $train_loss." )

            if has_test 
                println("Test KL Div. $test_KL_div | Testing acc. $test_acc.")#  | Testing MSE: $test_loss." )
                println("")
                println("Test conf: $conf.")
            end
        end
    end

   
    return TrainedMPS(W, MPSOptions(opts), EncodedTimeSeriesSet(training_states_meta)), training_information, EncodedTimeSeriesSet(testing_states_meta)

end
