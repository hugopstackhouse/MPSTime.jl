using Scratch
using JLD2


# save/load of TrainedMPS models with JLD2
rwpath = @get_scratch!("testing_readwrites")
fpath = joinpath(rwpath, "test.jld2")

f = JLD2.jldopen("Data/ecg200/datasets/ecg200.jld2", "r");
X_train = read(f, "X_train");
X_test = read(f, "X_test");
y_train = read(f, "y_train");
y_test = read(f, "y_test");
close(f)

# train MPS
opts = MPSOptions(d=4, chi_max=25;nsweeps=1, log_level=-5, verbosity=-10)
mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts);

# #TODO implement with scratch.jl
@save fpath mps
mps_old = mps
@load fpath mps
@test mps_old == mps # we are mainly concerned with serialisation issues here

# remove scratch space
delete_scratch!( @__MODULE__, "testing_readwrites")