module TestUtils

"""
    TestUtils.test_complex_ffts(ArrayType=Array; test_inplace=true, test_adjoint=true) 

Run tests to verify correctness of FFT, BFFT, and IFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_inplace=true`: whether to test in-place plans. 
- `test_adjoint=true`: whether to test [plan adjoints](api.md#Base.adjoint). 
"""
function test_complex_ffts end

"""
    TestUtils.test_real_ffts(ArrayType=Array; test_adjoint=true, copy_input=false)

Run tests to verify correctness of RFFT, BRFFT, and IRFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_adjoint=true`: whether to test [plan adjoints](api.md#Base.adjoint). 
- `copy_input=false`: whether to copy the input before applying the plan in tests, to accomodate for 
  [input-mutating behaviour of real FFTW plans](https://github.com/JuliaMath/AbstractFFTs.jl/issues/101).
"""
function test_real_ffts end

    # Always copy input before application due to FFTW real plans possibly mutating input (AbstractFFTs.jl#101)
"""
    TestUtils.test_plan(P::Plan, x::AbstractArray, x_transformed::AbstractArray;
                        inplace_plan=false, copy_input=false)

Test basic properties of a plan `P` given an input array `x` and expected output `x_transformed`.

Because [real FFTW plans may mutate their input in some cases](https://github.com/JuliaMath/AbstractFFTs.jl/issues/101), 
we allow specifying `copy_input=true` to allow for this behaviour in tests by copying the input before applying the plan.
"""
function test_plan end

"""
    TestUtils.test_plan_adjoint(P::Plan, x::AbstractArray; real_plan=false, copy_input=false)

Test basic properties of the [adjoint](api.md#Base.adjoint) `P'` of a particular plan given an input array `x`,
including its accuracy via the dot test. 

Real-to-complex and complex-to-real plans require a slightly modified dot test, in which case `real_plan=true` should be provided.
The plan is assumed out-of-place, as adjoints are not yet supported for in-place plans.
Because [real FFTW plans may mutate their input in some cases](https://github.com/JuliaMath/AbstractFFTs.jl/issues/101), 
we allow specifying `copy_input=true` to allow for this behaviour in tests by copying the input before applying the plan.
"""
function test_plan_adjoint end

if isdefined(Base, :get_extension) && isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Better error message if users forget to load Test
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            if any(f -> (f === exc.f), (test_real_ffts, test_complex_ffts, test_plan, test_plan_adjoint)) &&
                (Base.get_extension(AbstractFFTs, :AbstractFFTsTestExt) === nothing)
                print(io, "\nDid you forget to load Test?")
            end
        end
    end
end

end
