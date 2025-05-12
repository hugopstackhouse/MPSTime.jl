"""
The authors of this package wish to acknowledge the hard work of the UCR repository archivists, 
as well as the archivists and maintainers of timeseriesclassification.com and the aeon toolkit.
If you reuse this test data in your work, please make sure to attribute and acknowledge its
source!
"""


using Downloads, ZipFile, DelimitedFiles, JLD2
# download from timeseriesclassification.com
"""
Italy Power Demand:
The data was derived from twelve monthly electrical power demand
time series from Italy and first used in the paper "Intelligent
Icons: Integrating Lite-Weight Data Mining and Visualization into
GUI Operating Systems". The classification task is to distinguish
days from Oct to March (inclusive) from April to September.
"""

if ~isfile("Data/italypower/datasets/ipd.zip")
    Downloads.download("https://www.timeseriesclassification.com/aeon-toolkit/ItalyPowerDemand.zip", "Data/italypower/datasets/ipd.zip")
end


r = ZipFile.Reader("Data/italypower/datasets/ipd.zip")

raw_train, raw_test = nothing, nothing
for f in r.files
    if f.name == "ItalyPowerDemand_TRAIN.txt"
        # write("Data/ecg200/datasets")
        global raw_train = readdlm(f)
    elseif f.name ==  "ItalyPowerDemand_TEST.txt"
        global raw_test = readdlm(f)

    end
end

X_train = raw_train[:,2:end]
X_test = raw_test[:,2:end]

y_train::Vector{Int} = raw_train[:,1] .== 2
y_test::Vector{Int} = raw_test[:,1] .== 2

@save "Data/italypower/datasets/ItalyPowerDemandOrig.jld2" X_train y_train X_test y_test

"""
ECG200 Dataset:
This dataset was formatted by R. Olszewski as part of his thesis
“Generalized feature extraction for structural  pattern recognition
in time-series data,” at Carnegie Mellon University, 2001. Each
series traces the electrical activity recorded during one
heartbeat. The two classes are a normal heartbeat and a Myocardial
Infarction. 
"""

if ~isfile("Data/ecg200/datasets/ecg.zip")
    Downloads.download("https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip", "Data/ecg200/datasets/ecg.zip")
end



r = ZipFile.Reader("Data/ecg200/datasets/ecg.zip")

raw_train, raw_test = nothing, nothing
for f in r.files
    if f.name == "ECG200_TRAIN.txt"
        # write("Data/ecg200/datasets")
        global raw_train = readdlm(f)
    elseif f.name ==  "ECG200_TEST.txt"
        global raw_test = readdlm(f)

    end
end

X_train = raw_train[:,2:end]
X_test = raw_test[:,2:end]

y_train = (raw_train[:,1] .+1) ./2 .|>  Int
y_test = (raw_test[:,1] .+1) ./2 .|>  Int

@save "Data/ecg200/datasets/ecg200.jld2" X_train y_train X_test y_test
