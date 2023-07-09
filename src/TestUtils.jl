module TestUtils

"""
    TestUtils.test_complex_fft(ArrayType=Array; test_real=true, test_inplace=true) 

Run tests to verify correctness of FFT/BFFT/IFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_inplace=true`: whether to test in-place plans. 
- `test_adjoint=true`: whether to test [plan adjoints](api.md#Base.adjoint). 
"""
function test_complex_fft end

"""
    TestUtils.test_real_fft(ArrayType=Array; test_real=true, test_inplace=true)

Run tests to verify correctness of RFFT/BRFFT/IRFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_inplace=true`: whether to test in-place plans. 
- `test_adjoint=true`: whether to test [plan adjoints](api.md#Base.adjoint). 
"""
function test_real_fft end

"""
    TestUtils.test_plan_adjoint(P::Plan, x::AbstractArray; real_plan=false)

Test basic properties of the adjoint `P'` of a particular plan given an input array to the plan `x`,
including its accuracy via the dot test. Real-to-complex and complex-to-real plans require
a slightly modified dot test, in which case `real_plan=true` should be provided.

"""
function test_plan_adjoint end

function __init__()
    if isdefined(Base, :Experimental)
        # Better error message if users forget to load Test
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            if exc.f in (test_real_fft, test_complex_fft)
                print(io, "\nDid you forget to load Test?")
            end
        end
    end
end

end