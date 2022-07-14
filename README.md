# AbstractFFTs.jl

A general framework for fast Fourier transforms (FFTs) in Julia.

[![Travis](https://travis-ci.org/JuliaMath/AbstractFFTs.jl.svg?branch=master)](https://travis-ci.org/JuliaMath/AbstractFFTs.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaMath/AbstractFFTs.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMath/AbstractFFTs.jl?branch=master)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/AbstractFFTs.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/AbstractFFTs.jl/latest)

This package is mainly not intended to be used directly.
Instead, developers of packages that implement FFTs (such as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) or [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl))
extend the types/functions defined in `AbstractFFTs`.
This allows multiple FFT packages to co-exist with the same underlying `fft(x)` and `plan_fft(x)` interface.

## Developer information

To define a new FFT implementation in your own module, see [defining a new implementation](https://juliamath.github.io/AbstractFFTs.jl/stable/implementations/#Defining-a-new-implementation).
