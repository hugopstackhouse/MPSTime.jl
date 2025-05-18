contract_tensor = ITensors._contract

abstract type LossFunction <: Function end

abstract type KLDLoss <: LossFunction end
abstract type MSELoss <: LossFunction end

struct Loss_Grad_MSE <: MSELoss end
struct Loss_Grad_KLD <: KLDLoss end
struct Loss_Grad_KLD_slow <: KLDLoss end

struct Loss_Grad_mixed <: LossFunction end
struct Loss_Grad_default <: LossFunction end


loss_grad_MSE = Loss_Grad_MSE()
loss_grad_KLD = Loss_Grad_KLD()
loss_grad_KLD_slow = Loss_Grad_KLD_slow()

loss_grad_mixed = Loss_Grad_mixed()
loss_grad_default = Loss_Grad_default()



# Applying updates 

function custGD(
        tsep::TrainSeparate, 
        BT_init::BondTensor, 
        LE::PCache, 
        RE::PCache, 
        lid::Int, 
        rid::Int, 
        ETSs::EncodedTimeSeriesSet;
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
            println("Loss before step $i: $loss")
        end

    end

    return BT
end

function TSGO(
        tsep::TrainSeparate, 
        BT_init::BondTensor, 
        LE::PCache, 
        RE::PCache, 
        lid::Int, 
        rid::Int, 
        ETSs::EncodedTimeSeriesSet;
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
        
        @. BT -= eta * $/(grad, $norm(grad))  
        if verbosity >=1 && track_cost
            println("Loss before step $i: $loss")
        end

    end
    return BT
end

function apply_update(
        tsep::TrainSeparate, 
        BT_init::BondTensor, 
        LE::PCache, 
        RE::PCache, 
        lid::Int, 
        rid::Int,
        ETSs::EncodedTimeSeriesSet; 
        iters::Integer=10, 
        verbosity::Real=1, 
        dtype::DataType=ComplexF64, 
        loss_grad::Function=loss_grad_KLD, 
        bbopt::BBOpt,
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
        # if iscomplex
        #     C_index = Index(2, "C")
        #     bt_re = realise(BT_init, C_index; dtype=dtype)
        # else
        #     C_index = nothing
        #     bt_re = BT_init
        # end

        # if bbopt.name == "Optim" 
        #      # flatten bond tensor into a vector and get the indices
        #     bt_inds = inds(bt_re)
        #     bt_flat = NDTensors.array(bt_re, bt_inds) # should return a view

        #     # create anonymous function to feed into optim, function of bond tensor only
        #     fgcustom! = (F,G,B) -> loss_grad!(tsep, F, G, B, bt_inds, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
        #     # set the optimisation manfiold
        #     # apply optim using specified gradient descent algorithm and corresp. paramters 
        #     # set the manifold to either flat, sphere or Stiefel 
        #     if bbopt.fl == "CGD"
        #         method = Optim.ConjugateGradient(eta=eta)
        #     else
        #         method = Optim.GradientDescent(alphaguess=eta)
        #     end
        #     #method = Optim.LBFGS()
        #     res = Optim.optimize(Optim.only_fg!(fgcustom!), bt_flat; method=method, iterations = iters, 
        #     show_trace = (verbosity >=1),  g_abstol=1e-20)
        #     result_flattened = Optim.minimizer(res)

        #     BT_new = itensor(real(dtype), result_flattened, bt_inds)


        # elseif bbopt.name == "OptimKit"

        #     lg = BT -> loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
        #     if bbopt.fl == "CGD"
        #         alg = OptimKit.ConjugateGradient(; verbosity=verbosity, maxiter=iters)
        #     else
        #         alg = OptimKit.GradientDescent(; verbosity=verbosity, maxiter=iters)
        #     end
        #     BT_new, fx, _ = OptimKit.optimize(lg, bt_re, alg)


        if bbopt.name in ["Optim", "OptimKit"]
            error("Optim/OptimKit based solvers currently unimplemented for this version, set 'use_legacy_ITensor=true' in MPSOptions to enable")
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


# Highly Optimised Adding / product utilities for computing the KLD 

function kron_conj2(x1::Vector, x2::Vector)
    l1,l2 = length(x1),length(x2)
    out = Vector{eltype(x1)}(undef, l1*l2)
    @turbo for i = eachindex(x1), j=eachindex(x2)
        out[j + l2*(i-1)] = conj(x1[i] * x2[j])
    end
    return out
end


function kron_scaleadd_KLD!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, x1::Vector, x2::Vector)
    l2 = length(x2)
    yhat = 0.
    scale = yhat_p[]
    @turbo for i = eachindex(x1), j =eachindex(x2)
        idx = j + l2*(i-1)
        phi = conj(x1[i] * x2[j])
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function kron_scaleadd_firstsite_KLD!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yhat_p[]
    @turbo for i = eachindex(x1), j =eachindex(x2a)
        idx = j + l2*(i-1)
        phi = x1[i] * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function kron_scaleadd_KLD!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yhat_p[]
    @turbo for i = eachindex(x1), j =eachindex(x2a)
        idx = j + l2*(i-1)
        phi = conj(x1[i]) * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end


function kron_scaleadd_KLD!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector, x4::Vector)
    xa = kron_conj2(x1,x2)
    xb = kron_conj2(x3,x4)
    lb = length(xb)
    yhat = 0.
    scale = yhat_p[]
    @turbo for i = eachindex(xa), j = eachindex(xb)
        idx = j + lb*(i-1)
        phi = xa[i] * xb[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] / scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function yhat_phitilde_KLD!!(
        yhat::Base.RefValue{Float64},
        phi_tilde::AbstractVector,
        phit_prev::AbstractVector,
        bt::AbstractVector, 
        LEP::PCacheCol, 
        REP::PCacheCol, 
        product_state::PState, 
        lid::Int, 
        rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""

    ps = product_state.pstate

    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            @inbounds @fastmath kron_scaleadd_firstsite_KLD!(phi_tilde, phit_prev, bt, yhat, REP[rid+1], ps[rid],ps[lid])
        else
            @inbounds @fastmath kron_scaleadd_KLD!(phi_tilde, phit_prev, bt, yhat, ps[rid], ps[lid])
        end
    
    elseif rid == length(ps)
        # terminal site, no RE
        @inbounds @fastmath kron_scaleadd_KLD!(phi_tilde, phit_prev, bt, yhat, ps[rid], LEP[lid-1], ps[lid])

    else
        # we are in the bulk, both LE and RE exist
        @inbounds @fastmath kron_scaleadd_KLD!(phi_tilde, phit_prev, bt, yhat, REP[rid+1], ps[rid], LEP[lid-1], ps[lid])
    end
end



################################################################################################### KLD loss

function KLD_iter!( 
    yhat::Base.RefValue{Float64},
    phit_scaled::AbstractVector, 
    phit_prev::AbstractVector,
    BT_c::AbstractVector, 
    LEP::PCacheCol, 
    REP::PCacheCol,
    product_state::PState, 
    lid::Int, 
    rid::Int
    ) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""

    # it is assumed that BT has no label index, so yhat is a rank 0 tensor
    yhat_phitilde_KLD!!(yhat, phit_scaled, phit_prev, BT_c, LEP, REP, product_state, lid, rid)

    return -log(abs2(yhat[]))

end

function (::Loss_Grad_KLD)(
        ::TrainSeparate{false}, 
        bts::BondTensor, 
        LE::PCache, 
        RE::PCache,
        ETSs::EncodedTimeSeriesSet, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    # Assumes that the timeseries are sorted by class
 
    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    num_type = eltype(eltype(bts))
    # label_idx = inds(bts)[1]

    losses = zero(real(num_type))
    # grads = Tensor(zeros(num_type, size(bts)), inds(bts))
    phit_scaled = zeros(num_type, size(bts)) 
    yhat = Ref{Float64}(1.)
    phit_prev = Vector{num_type}(undef, size(bts,1)) 


 
    i_prev=0
    for (ci, cn) in enumerate(cnums)
        yhat[] = 1.
        phit_prev .= 0
        c_inds = (i_prev+1):(cn+i_prev)
        @inbounds @fastmath loss = mapreduce(
            (LEP,REP, prod_state) -> KLD_iter!( 
                yhat,
                view(phit_scaled, :,ci), 
                view(phit_prev, :),
                view(bts,:,ci), 
                LEP,
                REP,
                prod_state,
                lid,
                rid
            ),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])

        @inbounds @fastmath losses += loss
        @inbounds @fastmath @. phit_scaled[:, ci] = -conj(phit_scaled[:, ci] + phit_prev / yhat[] ) / $length(TSs)
        i_prev += cn
    end

    losses /= length(TSs)


    # @show phit_scaled


    return losses, phit_scaled

end



function (::Loss_Grad_KLD)(
        ::TrainSeparate{true}, 
        bts::BondTensor, 
        LE::PCache, 
        RE::PCache,
        ETSs::EncodedTimeSeriesSet, 
        lid::Int, 
        rid::Int
    )

    # Assumes that the timeseries are sorted by class

    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    num_type = eltype(eltype(bts))
    # label_idx = inds(bts)[1]

    losses = zero(real(num_type))
    # grads = Tensor(zeros(num_type, size(bts)), inds(bts))
    phit_scaled = zeros(num_type, size(bts)) 
    yhat = Ref{Float64}(1.)
    phit_prev = Vector{num_type}(undef, size(bts,1)) 

    i_prev=0
    for (ci, cn) in enumerate(cnums)
        yhat[] = 1.
        phit_prev .= 0
        c_inds = (i_prev+1):(cn+i_prev)
        @inbounds @fastmath loss = mapreduce(
            (LEP,REP, prod_state) -> KLD_iter!( 
                yhat,
                view(phit_scaled, :,ci), 
                view(phit_prev, :),
                view(bts,:,ci), 
                LEP,
                REP,
                prod_state,
                lid,
                rid
            ),+, eachcol(view(LE, :, c_inds)), eachcol(view(RE, :, c_inds)),TSs[c_inds])

        @inbounds @fastmath losses += loss / cn
        @inbounds @fastmath @. phit_scaled[:, ci] = -conj(phit_scaled[:, ci] + phit_prev / yhat[] ) / cn
        i_prev += cn
    end

    return losses, phit_scaled


end

#####################################################################################################  MSE LOSS
function kron_scaleadd_MSE!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, yprev_p::Base.RefValue{Float64}, x1::Vector, x2::Vector)
    l2 = length(x2)
    yhat = 0.
    scale = yhat_p[] - yprev_p[]
    @turbo for i = eachindex(x1), j = eachindex(x2)
        idx = j + l2*(i-1)
        phi = conj(x1[i] * x2[j])
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] * scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function kron_scaleadd_firstsite_MSE!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, yprev_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yhat_p[] - yprev_p[]
    @turbo for i = eachindex(x1), j = eachindex(x2a)
        idx = j + l2*(i-1)
        phi = x1[i] * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] * scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function kron_scaleadd_MSE!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, yprev_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector)
    x2a = kron_conj2(x2,x3)
    l2 = length(x2a)
    yhat = 0.
    scale = yhat_p[] - yprev_p[]
    @turbo for i = eachindex(x1), j = eachindex(x2a)
        idx = j + l2*(i-1)
        phi = conj(x1[i]) * x2a[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx] * scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end


function kron_scaleadd_MSE!(k::AbstractVector, kprev::AbstractVector, bt::AbstractVector, yhat_p::Base.RefValue{Float64}, yprev_p::Base.RefValue{Float64}, x1::Vector, x2::Vector, x3::Vector, x4::Vector)
    xa = kron_conj2(x1,x2)
    xb = kron_conj2(x3,x4)
    lb = length(xb)
    yhat = 0.
    scale = yhat_p[] - yprev_p[]

    
    @turbo for i = eachindex(xa), j = eachindex(xb)
        idx = j + lb*(i-1)
        phi = xa[i] * xb[j] 
        yhat += bt[idx] * phi
        k[idx] += kprev[idx]* scale
        kprev[idx] = phi
    end
    yhat_p[] = yhat
end

function yhat_phitilde_MSE!!(
        yhat::Base.RefValue{Float64},
        yprev::Base.RefValue{Float64},
        phi_tilde::AbstractVector,
        phit_prev::AbstractVector,
        bt::AbstractVector, 
        LEP::PCacheCol, 
        REP::PCacheCol, 
        product_state::PState, 
        lid::Int, 
        rid::Int
    )
    """Return yhat and phi_tilde for a bond tensor and a single product state"""

    ps = product_state.pstate

    if lid == 1
        if rid !== length(ps) # the fact that we didn't notice the previous version breaking for a two site MPS for nearly 5 months is hilarious
            # at the first site, no LE
            # formatted from left to right, so env - product state, product state - env
            @inbounds @fastmath kron_scaleadd_firstsite_MSE!(phi_tilde, phit_prev, bt, yhat, yprev, REP[rid+1], ps[rid], ps[lid])
        else
            @inbounds @fastmath kron_scaleadd_MSE!(phi_tilde, phit_prev, bt, yhat, yprev, ps[rid], ps[lid])
        end
    
    elseif rid == length(ps)
        # terminal site, no RE
        @inbounds @fastmath kron_scaleadd_MSE!(phi_tilde, phit_prev, bt, yhat, yprev, ps[rid], LEP[lid-1], ps[lid])

    else
        # we are in the bulk, both LE and RE exist
        @inbounds @fastmath kron_scaleadd_MSE!(phi_tilde, phit_prev, bt, yhat, yprev, REP[rid+1], ps[rid], LEP[lid-1], ps[lid])
    end
end



function MSE_iter!(
        yhat::Base.RefValue{Float64},
        yprev::Base.RefValue{Float64},
        phit_scaled::AbstractVector, 
        phit_prev::AbstractVector,
        BT_c::AbstractVector, 
        LEP::PCacheCol, 
        REP::PCacheCol,
        product_state::PState, 
        lid::Int, 
        rid::Int,
        mask::Float64
    
    ) 
    """Computes the Mean squared error loss function"""

    yhat_phitilde_MSE!!(yhat, yprev, phit_scaled, phit_prev, BT_c, LEP, REP, product_state, lid, rid)

    loss = 0.5 * abs2(yhat[]-mask)
    yprev[] = mask
    
    return loss

end


function (::Loss_Grad_MSE)(
        ::TrainSeparate{false}, 
        bts::BondTensor, 
        LE::PCache, 
        RE::PCache,
        ETSs::EncodedTimeSeriesSet, 
        lid::Int, 
        rid::Int
    )
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""

    cnums = ETSs.class_distribution
    TSs = ETSs.timeseries
    nts = length(TSs)
    num_type = eltype(eltype(bts))
    # label_idx = inds(bts)[1]

    losses = zero(real(num_type))
    # grads = Tensor(zeros(num_type, size(bts)), inds(bts))
    phit_scaled = zeros(num_type, size(bts)) 
    yhat = Ref{Float64}(1.)
    yprev = Ref{Float64}(0.)
    phit_prev = Vector{num_type}(undef, size(bts,1)) 
    class_mask = zeros(Float64, nts)

    i_prev=0
    for (ci, cn) in enumerate(cnums)
        yhat[] = 1.
        phit_prev .= 0
        c_inds = (i_prev+1):(cn+i_prev)
        class_mask[c_inds] .= 1.

        @inbounds @fastmath loss = mapreduce(
            (LEP, REP, prod_state, mask) -> MSE_iter!( 
                yhat,
                yprev,
                view(phit_scaled, :, ci), 
                view(phit_prev, :),
                view(bts, :, ci), 
                LEP,
                REP,
                prod_state,
                lid,
                rid,
                mask
            ),+, eachcol(LE), eachcol(RE), TSs, class_mask)

        @inbounds @fastmath losses += loss 
        @inbounds @fastmath @. phit_scaled[:, ci] = (phit_scaled[:, ci] + conj(phit_prev) * (yhat[] - yprev[])) / nts
        class_mask[c_inds] .= 0.
        i_prev += cn
    end

    return losses/nts, phit_scaled



end

#TODO implement mixed loss for array based learning
# ###################################################################################################  Mixed loss


# function mixed_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
#     product_state::PState, lid::Int, rid::Int; alpha=5) 
#     """Returns the loss and gradient that results from mixing the logarithmic loss and mean squared error loss with mixing parameter alpha"""

#     yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

#     # convert the label to ITensor
#     label_idx = inds(yhat)[1]
#     y = onehot(label_idx => (product_state.label_index))
#     f_ln = (yhat *y)[1]
#     log_loss = -log(abs2(f_ln))

#     # construct the gradient - return dC/dB
#     log_gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

#     # MSE
#     diff_sq = abs2.(yhat - y)
#     sum_of_sq_diff = sum(diff_sq)
#     MSE_loss = 0.5 * real(sum_of_sq_diff)

#     # construct the gradient - return dC/dB
#     MSE_gradient = (yhat - y) * conj(phi_tilde)


#     return [log_loss + alpha*MSE_loss, log_gradient + alpha*MSE_gradient]

# end


# function (::Loss_Grad_mixed)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int; alpha=5)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     TSs = ETSs.timeseries
#     loss,grad = mapreduce((LEP,REP, prod_state) -> mixed_iter(BT,LEP,REP,prod_state,lid,rid; alpha=alpha),+, eachcol(LE), eachcol(RE),TSs)
    
#     loss /= length(TSs)
#     grad ./= length(TSs)

#     return loss, grad

# end


# ######################### old  generic Loss_Grad function
# function (::Loss_Grad_default)(::TrainSeparate{false}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     TSs = ETSs.timeseries
#     loss,grad = mapreduce((LEP,REP, prod_state) -> lg_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
#     loss /= length(TSs)
#     grad ./= length(TSs)

#     return loss, grad

# end

# function (::Loss_Grad_default)(::TrainSeparate{true}, BT::ITensor, LE::PCache, RE::PCache,
#     ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int)
#     """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
#         Allows the input to be complex if that is supported by lg_iter"""
#     # Assumes that the timeseries are sorted by class
 
#     cnums = ETSs.class_distribution
#     TSs = ETSs.timeseries
#     label_idx = find_index(BT, "f(x)")

#     losses = ITensor(real(eltype(BT)), label_idx)
#     grads = ITensor(eltype(BT), inds(BT))

#     i_prev=0
#     for (ci, cn) in enumerate(cnums)
#         y = onehot(label_idx => ci)

#         c_inds = (i_prev+1):cn
#         loss, grad = mapreduce((LEP,REP, prod_state) -> KLD_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE)[c_inds], eachcol(RE)[c_inds],TSs[c_inds])

#         losses += loss  / cn # maybe doing this with a combiner instead will be more efficient
#         grads += grad / cn
#         i_prev = cn
#     end


#     return losses, grads

# end
