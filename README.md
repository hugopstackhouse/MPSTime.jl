<p align="center">
  <picture>
    <source srcset="docs/src/assets/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="docs/src/assets/logo.svg" alt="MPSTime logo" height="200"/>
  </picture>
</p>

<h1 align="center"><em>MPSTime.jl</em>: Matrix-Product States for Time-Series Analysis</h1>

<p align="center">
  <a href="https://doi.org/10.1103/61h8-8qr5"><img src="https://img.shields.io/badge/paper-Phys%20Rev%20Research%20(2025)-informational" alt="Paper"/></a>
  <a href="https://hugopstackhouse.github.io/MPSTime.jl/dev/"><img src="https://img.shields.io/badge/docs-latest-2ea44f" alt="Documentation"/></a>
  <a href="https://github.com/hugopstackhouse/MPSTime.jl/releases"><img src="https://img.shields.io/github/v/release/hugopstackhouse/MPSTime.jl?label=release" alt="Latest release"/></a>
  <a href="https://github.com/hugopstackhouse/MPSTime.jl/actions/workflows/CI.yml"><img src="https://github.com/hugopstackhouse/MPSTime.jl/actions/workflows/CI.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/JuliaTesting/Aqua.jl"><img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua QA"/></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License: GPL v3"/></a>
</p>

_MPSTime_ is a Julia package for learning the joint probability distribution of time series directly from data, using matrix-product state (MPS) methods inspired by quantum many-body physics.
It provides a unified formalism for classifying unseen data and for imputing gaps in time-series data, which regularly occur in real-world datasets due to sensor failure, routine maintenance, or other problems.
The package is built on [ITensors.jl](https://github.com/ITensor/ITensors.jl) and is described in [Moore et al. (2025)](https://doi.org/10.1103/61h8-8qr5).

**Full documentation:** [hugopstackhouse.github.io/MPSTime.jl](https://hugopstackhouse.github.io/MPSTime.jl/dev/), with tutorials for classification and imputation.

## Installation

_MPSTime_ requires Julia 1.11.1 or later and can be installed from the Julia General registry using the package manager:

```julia
julia> ]
pkg> add MPSTime
```

or, equivalently, `using Pkg; Pkg.add("MPSTime")`.

## Quick start

Fit an MPS to a small two-class demonstration dataset and classify held-out series (see the [classification tutorial](https://hugopstackhouse.github.io/MPSTime.jl/dev/classification/) for a full walkthrough):

```julia
using MPSTime, Random

rng = Xoshiro(1)
# two classes of noisy sinusoids with different periods: 20 training and 10 test series per class
X_train = vcat(
    trendy_sine(30, 20; sigma=0.1, slope=[-3,0,3], period=(12,15), rng=rng)[1],
    trendy_sine(30, 20; sigma=0.1, slope=[-3,0,3], period=(16,19), rng=rng)[1]
)
y_train = vcat(fill(1, 20), fill(2, 20))
X_test = vcat(
    trendy_sine(30, 10; sigma=0.2, slope=[-3,0,3], period=(12,15), rng=rng)[1],
    trendy_sine(30, 10; sigma=0.2, slope=[-3,0,3], period=(16,19), rng=rng)[1]
)
y_test = vcat(fill(1, 10), fill(2, 10))

opts = MPSOptions(nsweeps=1)  # other hyperparameters at their defaults
mps, info, test_states = fitMPS(X_train, y_train, X_test, y_test, opts)
predictions = classify(mps, X_test)  # a vector of predicted class labels
```

This takes about half a minute after compilation.
Use more sweeps and larger datasets for real problems; see the [hyperparameters guide](https://hugopstackhouse.github.io/MPSTime.jl/dev/hyperparameters/).

## Documentation

| Topic | Page |
|---|---|
| Classification | [tutorial](https://hugopstackhouse.github.io/MPSTime.jl/dev/classification/) |
| Imputation | [tutorial](https://hugopstackhouse.github.io/MPSTime.jl/dev/imputation/) |
| Encodings | [guide](https://hugopstackhouse.github.io/MPSTime.jl/dev/encodings/) |
| Hyperparameters | [guide](https://hugopstackhouse.github.io/MPSTime.jl/dev/hyperparameters/) |

## Citation

If you use this software in your work, please read and cite the [published paper](https://doi.org/10.1103/61h8-8qr5):

> Moore, J. B., Stackhouse, H. P., Fulcher, B. D. & Mahmoodian, S. (2025). Using matrix-product states for time-series machine learning. _Physical Review Research_ **7**, 043010. [doi:10.1103/61h8-8qr5](https://doi.org/10.1103/61h8-8qr5)

```bibtex
@article{MPSTime2025,
  title     = {Using matrix-product states for time-series machine learning},
  author    = {Moore, Joshua B. and Stackhouse, Hugo P. and Fulcher, Ben D. and Mahmoodian, Sahand},
  journal   = {Physical Review Research},
  volume    = {7},
  issue     = {4},
  pages     = {043010},
  numpages  = {31},
  year      = {2025},
  month     = {Oct},
  publisher = {American Physical Society},
  doi       = {10.1103/61h8-8qr5},
  url       = {https://link.aps.org/doi/10.1103/61h8-8qr5}
}
```

## Contributing

We welcome contributions from the community!
_MPSTime_ is under active development and research, making it an exciting opportunity for collaboration.
Whether you are interested in adding new features, improving existing documentation, fixing bugs, or sharing ideas, your input is valuable.
Feel free to submit a [PR](https://github.com/hugopstackhouse/MPSTime.jl/pulls) or open an [issue](https://github.com/hugopstackhouse/MPSTime.jl/issues) for discussion.

## License

_MPSTime_ is released under the [GNU General Public License v3.0](LICENSE).
