<p align="center">
  <picture>
    <source srcset="docs/src/assets/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="docs/src/assets/logo.svg" alt="mpstime logo" height="200"/>
  </picture>
</p>


<h1 align="center"><em>MPSTime.jl</em>: Matrix-Product States for Time-Series Analysis</h1>


<p align="center">
  <a href="https://hugopstackhouse.github.io/MPSTime.jl/dev/">
    <img src="https://img.shields.io/badge/docs-latest-2ea44f?" alt="docs - latest">
  </a>
    <img src="https://img.shields.io/badge/version-0.1.0--DEV-blue?" alt="version - 0.1.0-DEV">
  </a>
    <img src="https://github.com/hugopstackhouse/MPSTime.jl/actions/workflows/CI.yml/badge.svg">
  </a>
  <a href="https://github.com/JuliaTesting/Aqua.jl">
    <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg">
</p>

__MPSTime__ is a Julia package for learning the joint probability distribution of time series directly from data using matrix-product state (MPS) methods inspired by quantum many-body physics. 
It provides a unified formalism for classifying unseen data, as well as imputing gaps in time-series data, which regularly occur in real-world datasets due to sensor failure, routine maintenance, or other problems.

## Installation
This is not yet a registered Julia package, but it will be soon (TM)! In the meantime, you can install it directly from github:

```Julia
julia> ]
pkg> add https://github.com/hugopstackhouse/MPSTime.jl.git
```

## Usage and Documentation
We provide [tutorials](https://hugopstackhouse.github.io/MPSTime.jl/dev/classification/) and basic usage examples in the documentation:
- [LATEST DOCS](https://hugopstackhouse.github.io/MPSTime.jl/) -- the latest documentation.

## Citation
If you use this software in your work, please read and cite the [arXiv preprint](https://arxiv.org/abs/2412.15826):
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

## Contributing to MPSTime
We welcome contributions from the community! 
MPSTime is under active development and resarch, making it an exciting opprtunity for collaboration.
Whether you are interested in adding new features, improving existing documentation, fixing bugs, or sharing ideas, your input is valuable.
Feel free to submit a [PR](https://github.com/hugopstackhouse/MPSTime.jl/pulls) or open an [issue](https://github.com/hugopstackhouse/MPSTime.jl/issues) for discussion.
