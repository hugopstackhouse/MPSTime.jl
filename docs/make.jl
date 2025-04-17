using Documenter
using DocumenterCitations
using MPSTime

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(
    sitename = "MPSTime",
    format = Documenter.HTML(sidebar_sitename=false, assets=String["assets/citations.css"]),
    modules = [MPSTime],
    plugins = [bib],
    pages = [
    "Introduction" => "index.md",
    "Classification" => "classification.md",
    "Imputation" => "imputation.md",
    "Synthetic Data Generation" => "synthdatagen.md",
    "Encodings" => "encodings.md",
    "Hyperparameter Tuning" => "hyperparamaters.md",
    "Tools" => "tools.md",
    "Docstrings" => "docstrings.md",
    "References" => "references.md",
    ]
)

# deploydocs(
#     repo = "github.com/joshuabmoore/MPSTime.jl.git",
# )

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

# testing locally
# julia --color=yes --project make.jl

# julia -e 'using LiveServer; serve(dir="docs/build")'
