contract_tensor = ITensors._contract
⊗(t1::Tensor, t2::Tensor) = contract_tensor(t1, t2)

##### apply_update_IT
function realise(
        B::ITensor, 
        C_index::Index{Int64}; 
        dtype::DataType=ComplexF64
    )
    """Converts a Complex {s} dimension r itensor into a eal 2x{s} dimension itensor. Increases the rank from rank{s} to 1+ rank{s} by adding a 2-dimensional index "C_index" to the start"""
    ib = inds(B)
    inds_c = C_index,ib
    B_m = Array{dtype}(B, ib)

    out = Array{real(dtype)}(undef, 2,size(B)...)
    
    ls = eachslice(out; dims=1)
    
    ls[1] = real(B_m)
    ls[2] = imag(B_m)

    return ITensor(real(dtype), out, inds_c)
end


function complexify(
        B::ITensor, 
        C_index::Index{Int64}; 
        dtype::DataType=ComplexF64
    )
    """Converts a real 2x{s} dimension itensor into a Complex {s} dimension itensor. Reduces the rank from rank{s}+1 to rank{s} by removing the first index"""
    ib = inds(B)
    C_index, c_inds... = ib
    B_ra = NDTensors.array(B, ib) # should return a view


    re_part = selectdim(B_ra, 1,1);
    im_part = selectdim(B_ra, 1,2);

    return ITensor(dtype, complex.(re_part,im_part), c_inds)
end


function loss_grad_enforce_real(
        tsep::TrainSeparate, 
        BT::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int, 
        C_index::Union{Index{Int64},Nothing}; 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD
    )
    """Function for computing the loss function and the gradient over all samples using a left and right cache. 
        Takes a real itensor and will convert it to complex before calling loss_grad if dtype is complex. Returns a real gradient. """
    

    if isnothing(C_index) # the itensor is real
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
    else
        # pass in a complex itensor
        BT_c = complexify(BT, C_index; dtype=dtype)

        loss, grad = loss_grad(tsep, BT_c, LE, RE, ETSs, lid, rid)

        grad = realise(grad, C_index; dtype=dtype)
    end


    return loss, grad

end

function loss_grad!(
        tsep::TrainSeparate, 
        F,
        G,
        B_flat::AbstractArray,
        b_inds::Tuple{Vararg{Index{Int64}}}, 
        LE::PCacheIT, 
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int, 
        C_index::Union{Index{Int64},Nothing}; 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD
    )

    """Calculates the loss and gradient in a way compatible with Optim. Takes a flat, real array and converts it into an itensor before it passes it loss_grad """
    BT = itensor(real(dtype), B_flat, b_inds) # convert the bond tensor from a flat array to an itensor

    loss, grad = loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)

    if !isnothing(G)
        G .= NDTensors.array(grad,b_inds)
    end

    if !isnothing(F)
        return loss
    end

end


function custGD(
        tsep::TrainSeparate, 
        BT_init::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT, 
        lid::Int, 
        rid::Int, 
        ETSs::EncodedTimeSeriesSetIT;
        iters=10, 
        verbosity::Real=1, 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD, 
        track_cost::Bool=false, 
        eta::Real=0.01
    )
    BT = copy(BT_init)

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
        #zygote_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
        # update the bond tensor
        @. BT -= eta * grad
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

    end

    return BT
end

function TSGO(
        tsep::TrainSeparate, 
        BT_init::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT, 
        lid::Int, 
        rid::Int, 
        ETSs::EncodedTimeSeriesSetIT;
        iters=10, 
        verbosity::Real=1, 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD, 
        track_cost::Bool=false, 
        eta::Real=0.01
    )
    BT = copy(BT_init) # perhaps not necessary?
    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
        
        @. BT -= eta * $/(grad, $norm(grad)) #TODO investigate the absolutely bizarre behaviour that happens here with bigfloats if the arithmetic order is changed
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

    end
    return BT
end


function apply_update_IT(
        tsep::TrainSeparate, 
        BT_init::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT, 
        lid::Int, 
        rid::Int,
        ETSs::EncodedTimeSeriesSetIT; 
        iters::Integer=10, 
        verbosity::Real=1, 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD, 
        bbopt::BBOpt=BBOpt("Optim"),
        track_cost::Bool=false, 
        eta::Real=0.01, 
        rescale::Tuple{Bool,Bool} = (false, true)
    )
    """Apply update to bond tensor using the method specified by BBOpt. Will normalise B before and/or after it computes the update B+dB depending on the value of rescale [before::Bool,after::Bool]"""

    iscomplex = !(dtype <: Real)

    if rescale[1]
        normalize!(BT_init)
    end

    if bbopt.name == "CustomGD"
        if uppercase(bbopt.fl) == "GD"
            BT_new = custGD(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

        elseif uppercase(bbopt.fl) == "TSGO"
            BT_new = TSGO(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

        end
    else
        # break down the bond tensor to feed into optimkit or optim
        if iscomplex
            C_index = Index(2, "C")
            bt_re = realise(BT_init, C_index; dtype=dtype)
        else
            C_index = nothing
            bt_re = BT_init
        end

        if bbopt.name == "Optim" 
             # flatten bond tensor into a vector and get the indices
            bt_inds = inds(bt_re)
            bt_flat = NDTensors.array(bt_re, bt_inds) # should return a view

            # create anonymous function to feed into optim, function of bond tensor only
            fgcustom! = (F,G,B) -> loss_grad!(tsep, F, G, B, bt_inds, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
            # set the optimisation manfiold
            # apply optim using specified gradient descent algorithm and corresp. paramters 
            # set the manifold to either flat, sphere or Stiefel 
            if bbopt.fl == "CGD"
                method = Optim.ConjugateGradient(eta=eta)
            else
                method = Optim.GradientDescent(alphaguess=eta)
            end
            #method = Optim.LBFGS()
            res = Optim.optimize(Optim.only_fg!(fgcustom!), bt_flat; method=method, iterations = iters, 
            show_trace = (verbosity >=1),  g_abstol=1e-20)
            result_flattened = Optim.minimizer(res)

            BT_new = itensor(real(dtype), result_flattened, bt_inds)


        elseif bbopt.name == "OptimKit"

            lg = BT -> loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
            if bbopt.fl == "CGD"
                alg = OptimKit.ConjugateGradient(; verbosity=verbosity, maxiter=iters)
            else
                alg = OptimKit.GradientDescent(; verbosity=verbosity, maxiter=iters)
            end
            BT_new, fx, _ = OptimKit.optimize(lg, bt_re, alg)


        else
            error("Unknown Black Box Optimiser $bbopt, options are [CustomGD, Optim, OptimKit]")
        end

        if iscomplex # convert back to a complex itensor
            BT_new = complexify(BT_new, C_index; dtype=dtype)
        end
    end

    if rescale[2]
        normalize!(BT_new)
    end

    if track_cost
        loss, grad = loss_grad(tsep, BT_new, LE, RE, ETSs, lid, rid)
        println("Loss at site $lid*$rid: $loss")
    end

    return BT_new

end


####################################################### yhat_phitilde helper functions

function yhat_phitilde_left(
        BT::Tensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT, 
        product_state::PStateIT, 
        lid::Int, rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""

    ps = product_state.pstate

    psl = tensor(ps[lid])
    psr = tensor(ps[rid])
    

    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            rc = tensor(REP[rid+1])
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            # @show inds(phi_tilde)
            # @show inds(conj.(psl⊗psr) ⊗ rc)
            @inbounds @fastmath phi_tilde =  conj.(psr) ⊗ rc ⊗ conj.(psl)
        end
       
    elseif rid == length(ps)
        lc = tensor(LEP[lid-1])
    
        # terminal site, no RE
        # temp = $⊗(conj($⊗(psr ⊗ psl)),
        # lc)

        # @show inds(phi_tilde)
        # @show inds(temp)
        @inbounds @fastmath phi_tilde =  conj(psr ⊗ psl) ⊗ lc

    else
        rc = tensor(REP[rid+1])
        lc = tensor(LEP[lid-1])

        
        # tmp = ⊗(⊗(⊗(conj(psr), rc), 
        # conj(psl)), lc )

        # @show inds(phi_tilde)
        # @show inds(tmp)
        @inbounds @fastmath phi_tilde =  conj(psr) ⊗ rc ⊗ conj(psl) ⊗ lc

    end

    # if inds(BT) !== inds(phi_tilde)
    #     @show(inds(BT))
    #     @show(inds(phi_tilde))
    # end


    @inbounds @fastmath yhat = BT ⊗ phi_tilde # NOT a complex inner product !! 

    return yhat, phi_tilde

end

function yhat_phitilde_right(
        BT::Tensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT, 
        product_state::PStateIT, 
        lid::Int, 
        rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""

    ps = product_state.pstate

    psl = tensor(ps[lid])
    psr = tensor(ps[rid])
    

    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            @inbounds rc = tensor(REP[rid+1])
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            # @show inds(phi_tilde)
            @inbounds @fastmath phi_tilde =  conj(psl ⊗ psr) ⊗ rc
        end
       
    elseif rid == length(ps)
        @inbounds lc = tensor(LEP[lid-1])
    
        # terminal site, no RE
        @inbounds @fastmath phi_tilde = conj(psl) ⊗ lc ⊗ conj(psr)

    else
        @inbounds rc = tensor(REP[rid+1])
        @inbounds lc = tensor(LEP[lid-1])
        # going right
        @inbounds @fastmath phi_tilde = conj(psl) ⊗ lc ⊗ conj(psr) ⊗ rc 

        # we are in the bulk, both LE and RE exist
        # phi_tilde ⊗= LEP[lid-1] ⊗ REP[rid+1]

    end

    # if inds(BT) !== inds(phi_tilde)
    #     @show(inds(BT))
    #     @show(inds(phi_tilde))
    # end

    @inbounds @fastmath yhat = BT ⊗ phi_tilde # NOT a complex inner product !! 

    return yhat, phi_tilde

end

function yhat_phitilde(
        BT::Tensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT, 
        product_state::PStateIT,
        lid::Int, 
        rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    if hastags(ind(BT, 1), "Site,n=$lid")
        return yhat_phitilde_right(
            BT, 
            LEP, 
            REP, 
            product_state, 
            lid, 
            rid
        )
    else
        return yhat_phitilde_left(
            BT, 
            LEP, 
            REP, 
            product_state, 
            lid, 
            rid
        )
    end
end

function yhat_phitilde!(
        phi_tilde::ITensor, 
        BT::ITensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT, 
        product_state::PStateIT, 
        lid::Int, 
        rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""


    ps = product_state.pstate

    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            @inbounds @fastmath phi_tilde .=  conj.(ps[lid] * ps[rid]) * REP[rid+1]
        end
       
    elseif rid == length(ps)
        # terminal site, no RE
        phi_tilde .=  conj.(ps[rid] * ps[lid]) * LEP[lid-1] 

    else
        if hastags(ind(BT, 1), "Site,n=$lid")
            # going right
            @inbounds @fastmath phi_tilde .= conj.(ps[lid]) * LEP[lid-1] * conj.(ps[rid]) * REP[rid+1]
        else
            # going left
            @inbounds @fastmath phi_tilde .=  conj.(ps[rid]) * REP[rid+1] * conj.(ps[lid]) * LEP[lid-1] 
        end
        # we are in the bulk, both LE and RE exist
        # phi_tilde *= LEP[lid-1] * REP[rid+1]

    end


    @inbounds @fastmath yhat = BT * phi_tilde # NOT a complex inner product !! 

    return yhat

end

################################################################################################### KLD loss



function KLD_iter!(
        phit_scaled::Tensor, 
        BT_c::Tensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT,
        product_state::PStateIT, 
        lid::Int, 
        rid::Int
    ) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""
    
    # it is assumed that BT has no label index, so yhat is a rank 0 tensor
    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    f_ln = yhat[1]
    loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    # gradient = -conj(phi_tilde / f_ln) 
    @inbounds @fastmath @. phit_scaled += phi_tilde / f_ln

    return loss

end


function (::Loss_Grad_KLD)(
        ::TrainSeparate{true}, 
        BT::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = inds(BT)[1]

    losses = zero(real(eltype(BT)))
    grads = Tensor(zeros(eltype(BT), size(BT)), inds(BT))
    no_label = inds(BT)[2:end]
    phit_scaled = Tensor(eltype(BT), no_label)
    # phi_tilde = Tensor(eltype(BT), no_label)


    i_prev = 0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)
        bt = tensor(BT * y)
        phit_scaled .= zero(eltype(bt))

        c_inds = (i_prev+1):(cn+i_prev)
        @inbounds @fastmath loss = mapreduce((LEP,REP, prod_state) -> KLD_iter!( phit_scaled,bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
        @inbounds @fastmath losses += loss/cn # maybe doing this with a combiner instead will be more efficient
        @inbounds @fastmath @. $selectdim(grads,1, ci) -= conj(phit_scaled) / cn

        i_prev += cn

    end


    return losses, itensor(eltype(BT), grads, inds(BT))

end

function (::Loss_Grad_KLD)(
        ::TrainSeparate{false}, 
        BT::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = inds(BT)[1]

    losses = zero(real(eltype(BT)))
    grads = Tensor(zeros(eltype(BT), size(BT)), inds(BT))
    no_label = inds(BT)[2:end]
    phit_scaled = Tensor(eltype(BT), no_label)
    # phi_tilde = Tensor(eltype(BT), no_label)


 
    i_prev=0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)
        bt = tensor(BT * y)
        phit_scaled .= zero(eltype(bt))

        c_inds = (i_prev+1):(cn+i_prev)
        @inbounds @fastmath loss = mapreduce((LEP,REP, prod_state) -> KLD_iter!( phit_scaled,bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
        @inbounds @fastmath losses += loss # maybe doing this with a combiner instead will be more efficient
        @inbounds @fastmath @. $selectdim(grads,1, ci) -= conj(phit_scaled)
        #### equivalent without mapreduce
        # for ci in c_inds 
        #     # mapreduce((LEP,REP, prod_state) -> KLD_iter(bt,LEP,REP,prod_state,lid,rid),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])
        #     # valid = map(ts -> ts.label_index == ci, TSs[c_inds]) |> all
        #     LEP = view(LE, :, ci)
        #     REP = view(RE, :, ci)
        #     prod_state = TSs[ci]
        #     loss, grad = KLD_iter(bt,LEP,REP,prod_state,lid,rid)

        #     losses += loss # maybe doing this with a combiner instead will be more efficient
        #     grads .+= grad * y 
        # end
        #####
        
        i_prev += cn
    end

    losses /= length(TSs)
    grads ./= length(TSs)


    return losses, itensor(eltype(BT), grads, inds(BT))

end
#####################################################################################################  MSE LOSS

function MSE_iter(
        BT_c::ITensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT,
        product_state::PStateIT, 
        lid::Int, 
        rid::Int
    ) 
    """Computes the Mean squared error loss function derived from KL divergence and its gradient"""


    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label_index))

    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    gradient = (yhat - y) * conj(phi_tilde)

    return [loss, gradient]

end


function (::Loss_Grad_MSE)(
        ::TrainSeparate{false}, 
        BT::ITensor, 
        LE::PCacheIT,
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> MSE_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end

###################################################################################################  Mixed loss


function mixed_iter(
        BT_c::ITensor, 
        LEP::PCacheColIT, 
        REP::PCacheColIT,
        product_state::PStateIT, 
        lid::Int, 
        rid::Int; 
        alpha=5
    ) 
    """Returns the loss and gradient that results from mixing the logarithmic loss and mean squared error loss with mixing parameter alpha"""

    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label_index))
    f_ln = (yhat *y)[1]
    log_loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    log_gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

    # MSE
    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    MSE_loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    MSE_gradient = (yhat - y) * conj(phi_tilde)


    return [log_loss + alpha*MSE_loss, log_gradient + alpha*MSE_gradient]

end


function (::Loss_Grad_mixed)(
        ::TrainSeparate{false}, 
        BT::ITensor, 
        LE::PCacheIT, 
        RE::PCacheIT,
        ETSs::EncodedTimeSeriesSetIT, 
        lid::Int, 
        rid::Int; 
        alpha=5
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> mixed_iter(BT,LEP,REP,prod_state,lid,rid; alpha=alpha),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end


######################### old  generic Loss_Grad function
function (::Loss_Grad_default)(::TrainSeparate{false}, BT::ITensor, LE::PCacheIT, RE::PCacheIT,
    ETSs::EncodedTimeSeriesSetIT, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    TSs = ETSs.timeseries
    loss,grad = mapreduce((LEP,REP, prod_state) -> lg_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end

function (::Loss_Grad_default)(::TrainSeparate{true}, BT::ITensor, LE::PCacheIT, RE::PCacheIT,
    ETSs::EncodedTimeSeriesSetIT, lid::Int, rid::Int)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    label_idx = find_index(BT, "f(x)")

    losses = ITensor(real(eltype(BT)), label_idx)
    grads = ITensor(eltype(BT), inds(BT))

    i_prev=0
    for (ci, cn) in enumerate(cnums)
        y = onehot(label_idx => ci)

        c_inds = (i_prev+1):cn
        loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE)[c_inds], eachcol(RE)[c_inds],TSs[c_inds])

        losses += loss  / cn # maybe doing this with a combiner instead will be more efficient
        grads += grad / cn
        i_prev = cn
    end


    return losses, grads

end
