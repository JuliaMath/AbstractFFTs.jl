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

To define a new FFT implementation in your own module, you should

* Define a new subtype (e.g. `MyPlan`) of `AbstractFFTs.Plan{T}` for FFTs and related transforms on arrays of `T`.
  This must have a `pinv::Plan` field, initially undefined when a `MyPlan` is created, that is used for caching the
  inverse plan.

* Define a new method `AbstractFFTs.plan_fft(x, region; kws...)` that returns a `MyPlan` for at least some types of
  `x` and some set of dimensions `region`.   The `region` (or a copy thereof) should be accessible via `fftdims(p::MyPlan)` (which defaults to `p.region`).

* Define a method of `LinearAlgebra.mul!(y, p::MyPlan, x)` (or `A_mul_B!(y, p::MyPlan, x)` on Julia prior to
  0.7.0-DEV.3204) that computes the transform `p` of `x` and stores the result in `y`.

* Define a method of `*(p::MyPlan, x)`, which can simply call your `mul!` (or `A_mul_B!`) method.
  This is not defined generically in this package due to subtleties that arise for in-place and real-input FFTs.

* If the inverse transform is implemented, you should also define `plan_inv(p::MyPlan)`, which should construct the
  inverse plan to `p`, and `plan_bfft(x, region; kws...)` for an unnormalized inverse ("backwards") transform of `x`.

* You can also define similar methods of `plan_rfft` and `plan_brfft` for real-input FFTs.

The normalization convention for your FFT should be that it computes $y_k = \sum_j \exp\(-2 \pi i \cdot \frac{j k}{n}\) x_j$
for a transform of length $n$, and the "backwards" (unnormalized inverse) transform computes the same thing but with
$\exp\(+2 \pi i \cdot \frac{j k}{n}\)$.
