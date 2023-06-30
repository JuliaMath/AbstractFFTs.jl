# AbstractFFTs.jl

A general framework for fast Fourier transforms (FFTs) in Julia.

[![GHA](https://github.com/JuliaMath/AbstractFFTs.jl/workflows/CI/badge.svg)](https://github.com/JuliaMath/AbstractFFTs.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Codecov](http://codecov.io/github/JuliaMath/AbstractFFTs.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaMath/AbstractFFTs.jl?branch=master)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/AbstractFFTs.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/AbstractFFTs.jl/dev)

This package is mainly not intended to be used directly.
Instead, developers of packages that implement FFTs (such as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) or [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl))
extend the types/functions defined in `AbstractFFTs`.
This allows multiple FFT packages to co-exist with the same underlying `fft(x)` and `plan_fft(x)` interface.

## Developer information

To define a new FFT implementation in your own module, see [defining a new implementation](https://juliamath.github.io/AbstractFFTs.jl/stable/implementations/#Defining-a-new-implementation).

