include("../../LogLoss/RealRealHighDimension.jl")
using JLD2

dloc =  "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)



setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = :legendre_no_norm #legendre(norm=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
d=12
chi_max=60
opts=MPSOptions(; nsweeps=12, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.0025, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, init_rng=4567, chi_init=4, log_level=0)



# saveMPS(W, "LogLoss/saved/loglossout.h5")
# print_opts(opts)




if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test, opts; test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test, opts; test_run=false)

    # print_opts(opts)
    # summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    # sweep_summary(info)
end

save = true
if save
    scaler = fit(RobustSigmoid, X_train)
    r = model_encoding(opts.encoding).range

    X_train_scaled, X_test_scaled, norms, oob_rescales = transform_data(permutedims(X_train), permutedims(X_test); opts=opts)

    svpath = "Data/italypower/mps_saves/legendreNN2_d$(d)_chi$(chi_max).jld2"
    f = jldopen(svpath, "w")
        write(f, "X_train_scaled", X_train_scaled)
        write(f, "y_train", y_train)
        write(f, "X_test_scaled", X_test_scaled)
        write(f, "y_test", y_test);
        write(f, "mps", W)
        write(f, "opts", opts)
    close(f)
end