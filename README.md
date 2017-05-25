# AbstractFFTs.jl

A general framework for fast Fourier transforms (FFTs) in Julia.

This package is mainly not intended to be used directly.  Instead, developers of packages that implement FFTs (such as [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)) extend the types/functions defined in `AbstractFFTs`.  This multiple FFT packages to co-exist with the same underlying `fft(x)` and `plan_fft(x)` interface.

## Developer information

To define a new FFT implementation in your own module, you should

* Define a new subtype (e.g. `MyPlan`) of `AbstractFFTs.Plan{T}` for FFTs and related transforms on arrays of `T`.  This must have a `pinv::Plan` field, initially undefined when a `MyPlan` is created, that is used for caching the inverse plan.

* Define a new method `AbstractFFTs.plan_fft(x, region; kws...)` that returns a `MyPlan` for at least some types of `x` and some set of dimensions `region`. 

* Define a method of `A_mul_B!(y, p::MyPlan, x)` that computes the transform `p` of `x` and stores the result in `y`.

* If the inverse transform is implemented, you should also define `plan_inv(p::MyPlan)`, which should construct the inverse plan to `p`, and `plan_bfft(x, region; kws...)` for an unnormalized inverse ("backwards") transform of `x`. 

* You can also define similar methods of `plan_rfft` and `plan_brfft` for real-input FFTs.

The normalization convention for your FFT should be that it computes yₖ = ∑ⱼ xⱼ exp(-2πi jk/n) for a transform of length n, and the "backwards" (unnormalized inverse) transform computes the same thing but with exp(+2πi jk/n).
