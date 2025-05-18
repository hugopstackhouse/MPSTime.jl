# initialisation

encoding_names = [:uniform, :stoudenmire, :legendre, :fourier, :legendre_norm, histogram_split(:fourier), uniform_split(:legendre), :sahand_legendre, :SLTD]


for enc in encoding_names
    model_enc = MPSTime.model_encoding(enc)
    @test MPSTime.model_encoding(enc) == MPSTime.model_encoding(MPSTime.symbolic_encoding(model_enc)) # can all the bases get encoded and inverted correctly
end

# demo_opts = MPSOptions(; d=10)

# projections


# unif_split and hist_split

# custom encoding
