include("hyperopt.jl")

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

Xs = [X_train ; X_val] 
ys = [y_train; y_val] 

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = Basis("Legendre")
encode_classes_separately = false
train_classes_separately = false


etas = [0.01,0.05]
max_sweeps=10
ds = [3]
chi_maxs=[10,20]
nfolds=1

results = hyperopt(encoding, Xs, ys; etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs, nfolds=nfolds, distribute=true, train_ratio=0.9)


#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? (-1,-1) : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :maxacc)
val, ind = findmax(val_accs)

f, swi, etai, di, chmi, ei = Tuple(ind)

println("Best acc $(val[1]) occured at:\nsweep=$(val[2])\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


