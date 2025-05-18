# MPSTime.jl
A Julia package for time-series machine learning (ML) using Matrix-Product States (MPS) built on the [ITensors.jl](https://github.com/ITensor/ITensors.jl) framework [ITensor, ITensor-r0.3](@cite).

![](./assets/logo.svg)


## Overview

__MPSTime__ is a Julia package for learning the joint probability distribution of time series directly from data using [matrix product state (MPS)](https://en.wikipedia.org/wiki/Matrix_product_state) methods inspired by quantum many-body physics. 
It provides a unified formalism for:
- Time-series classification (inferring the class of unseen time-series).
- Univariate time-series imputation (inferring missing points within time-series instances) across fixed-length time series.
- Synthetic data generation (coming soon).

!!! info "Info"
    MPSTime is currently under active development. Many features are in an experimental stage and may undergo significant changes, refinements, or removals.

## Installation
This is not yet a registered Julia package, but it will be soon (TM)! In the meantime, you can install it directly from our [GitHub repository](https://github.com/hugopstackhouse/MPSTime.jl.git):

```Julia
julia> ]
pkg> add https://github.com/hugopstackhouse/MPSTime.jl.git
```

## Usage
See the sidebars for basic usage examples. 
We're continually adding more features and documentation as we go.

## Citation
If you use MPSTime in your work, please read and cite the [arXiv preprint](https://arxiv.org/abs/2412.15826):
```
@misc{MPSTime2024,
      title={Using matrix-product states for time-series machine learning}, 
      author={Joshua B. Moore and Hugo P. Stackhouse and Ben D. Fulcher and Sahand Mahmoodian},
      year={2024},
      eprint={2412.15826},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2412.15826}, 
}
```
