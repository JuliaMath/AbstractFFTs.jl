# FFT Implementations

## Existing packages

The following packages extend the functionality provided by AbstractFFTs:

* [FFTW.jl](https://github.com/JuliaMath/FFTW.jl): Bindings for the
  [FFTW](http://www.fftw.org) library. This also used to be part of Base Julia.
* [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl):
  Pure-Julia implementation of FFT, with support for arbitrary AbstractFloat types.

## Defining a new implementation

To define a new FFT implementation in your own module, you should

* Define a new subtype (e.g. `MyPlan`) of `AbstractFFTs.Plan{T}` for FFTs and related transforms on arrays of `T`.
  This must have a `pinv::Plan` field, initially undefined when a `MyPlan` is created, that is used for caching the
  inverse plan.

* Define a new method `AbstractFFTs.plan_fft(x, region; kws...)` that returns a `MyPlan` for at least some types of
  `x` and some set of dimensions `region`.   The `region` (or a copy thereof) should be accessible via `fftdims(p::MyPlan)` (which defaults to `p.region`).

* Define a method of `LinearAlgebra.mul!(y, p::MyPlan, x)` that computes the transform `p` of `x` and stores the result in `y`.

* Define a method of `*(p::MyPlan, x)`, which can simply call your `mul!` method.
  This is not defined generically in this package due to subtleties that arise for in-place and real-input FFTs.

* If the inverse transform is implemented, you should also define `plan_inv(p::MyPlan)`, which should construct the
  inverse plan to `p`, and `plan_bfft(x, region; kws...)` for an unnormalized inverse ("backwards") transform of `x`.
  Implementations only need to provide the unnormalized backwards FFT, similar to FFTW, and we do the scaling generically
  to get the inverse FFT.

* You can also define similar methods of `plan_rfft` and `plan_brfft` for real-input FFTs.

* To enable automatic computation of adjoint plans via [`Base.adjoint`](@ref) (used in rules for reverse-mode differentiation), define the trait `AbstractFFTs.ProjectionStyle(::MyPlan)`, which can return:
    * `AbstractFFTs.NoProjectionStyle()`,
    * `AbstractFFTs.RealProjectionStyle()`, for plans that halve one of the output's dimensions analogously to [`rfft`](@ref),
    * `AbstractFFTs.RealInverseProjectionStyle(d::Int)`, for plans that expect an input with a halved dimension analogously to [`irfft`](@ref), where `d` is the original length of the dimension.

The normalization convention for your FFT should be that it computes ``y_k = \sum_j x_j \exp(-2\pi i j k/n)`` for a transform of
length ``n``, and the "backwards" (unnormalized inverse) transform computes the same thing but with ``\exp(+2\pi i jk/n)``.
