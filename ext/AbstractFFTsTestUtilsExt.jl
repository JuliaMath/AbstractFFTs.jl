# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

module AbstractFFTsTestUtilsExt

using AbstractFFTs
using AbstractFFTs: TestUtils
using AbstractFFTs.LinearAlgebra
using Test

# Ground truth _x_fft computed using FFTW library
const TEST_CASES = (
            (; x = collect(1:7), dims = 1,
             x_fft = [28.0 + 0.0im,
                          -3.5 + 7.267824888003178im,
                          -3.5 + 2.7911568610884143im,
                          -3.5 + 0.7988521603655248im,
                          -3.5 - 0.7988521603655248im,
                          -3.5 - 2.7911568610884143im,
                          -3.5 - 7.267824888003178im]),
            (; x = collect(1:8), dims = 1,
             x_fft = [36.0 + 0.0im,
                          -4.0 + 9.65685424949238im,
                          -4.0 + 4.0im,
                          -4.0 + 1.6568542494923806im,
                          -4.0 + 0.0im,
                          -4.0 - 1.6568542494923806im,
                          -4.0 - 4.0im,
                          -4.0 - 9.65685424949238im]),
            (; x = collect(reshape(1:8, 2, 4)), dims = 2,
             x_fft = [16.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im;
                          20.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im]),
            (; x = collect(reshape(1:9, 3, 3)), dims = 2,
             x_fft = [12.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
                          15.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
                          18.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im]),
            (; x = collect(reshape(1:8, 2, 2, 2)), dims = 1:2,
             x_fft = cat([10.0 + 0.0im -4.0 + 0.0im; -2.0 + 0.0im 0.0 + 0.0im],
                             [26.0 + 0.0im -4.0 + 0.0im; -2.0 + 0.0im 0.0 + 0.0im],
                             dims=3)),
            (; x = collect(1:7) + im * collect(8:14), dims = 1,
             x_fft = [28.0 + 77.0im,
                          -10.76782488800318 + 3.767824888003175im,
                          -6.291156861088416 - 0.7088431389115883im,
                          -4.298852160365525 - 2.7011478396344746im,
                          -2.7011478396344764 - 4.298852160365524im,
                          -0.7088431389115866 - 6.291156861088417im,
                          3.767824888003177 - 10.76782488800318im]),
            (; x = collect(reshape(1:8, 2, 2, 2)) + im * reshape(9:16, 2, 2, 2), dims = 1:2,
             x_fft = cat([10.0 + 42.0im -4.0 - 4.0im; -2.0 - 2.0im 0.0 + 0.0im],
                             [26.0 + 58.0im -4.0 - 4.0im; -2.0 - 2.0im 0.0 + 0.0im],
                             dims=3)),
        )

# Perform generic adjoint plan tests 
function _adjoint_test(P, x; real_plan=false)
    y = rand(eltype(P * x), size(P * x))
    # test basic properties
    @test_broken eltype(P') === typeof(y) # (AbstactFFTs.jl#110)
    @test fftdims(P') == fftdims(P)
    @test (P')' === P # test adjoint of adjoint
    @test size(P') == AbstractFFTs.output_size(P) # test size of adjoint 
    # test correctness of adjoint and its inverse via the dot test
    if !real_plan
        @test dot(y, P * x) ≈ dot(P' * y, x)
        @test dot(y, P \ x) ≈ dot(P' \ y, x) 
    else
        _component_dot(x, y) = dot(real.(x), real.(y)) + dot(imag.(x), imag.(y))
        @test _component_dot(y, P * copy(x)) ≈ _component_dot(P' * copy(y), x)
        @test _component_dot(x, P \ copy(y)) ≈ _component_dot(P' \ copy(x), y) 
    end
    @test_throws MethodError mul!(x, P', y)
end

"""
    TestUtils.test_complex_fft(ArrayType=Array; test_real=true, test_inplace=true) 

Run tests to verify correctness of FFT/BFFT/IFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_inplace=true`: whether to test in-place plans. 
- `test_adjoint=true`: whether to test adjoints of plans. 
"""
function TestUtils.test_complex_fft(ArrayType=Array; test_inplace=true, test_adjoint=true) 
    @testset "correctness of fft, bfft, ifft" begin
        for test_case in TEST_CASES
            _x, dims, _x_fft = test_case.x, test_case.dims, test_case.x_fft
            x = convert(ArrayType, _x) # dummy array that will be passed to plans
            x_complexf = convert(ArrayType, complex.(float.(x))) # for testing mutating complex FFTs
            x_fft = convert(ArrayType, _x_fft)

            # FFT
            @test fft(x, dims) ≈ x_fft
            if test_inplace
                _x_complexf = copy(x_complexf)
                @test fft!(_x_complexf, dims) ≈ x_fft
                @test _x_complexf ≈ x_fft
            end
            # test OOP plans, checking plan_fft and also inv of plan_ifft, 
            # which should give functionally identical plans
            for P in (plan_fft(similar(x_complexf), dims), inv(plan_ifft(similar(x_complexf), dims)))
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                @test P * x ≈ x_fft
                @test P \ (P * x) ≈ x
                _x_out = similar(x_fft)
                @test mul!(_x_out, P, x_complexf) ≈ x_fft
                @test _x_out ≈ x_fft
                if test_adjoint
                    _adjoint_test(P, x_complexf)
                end
            end
            if test_inplace
                # test IIP plans
                for P in (plan_fft!(similar(x_complexf), dims), inv(plan_ifft!(similar(x_complexf), dims)))
                    @test eltype(P) <: Complex
                    @test fftdims(P) == dims
                    _x_complexf = copy(x_complexf)
                    @test P * _x_complexf ≈ x_fft
                    @test _x_complexf ≈ x_fft
                    @test P \ _x_complexf ≈ x
                    @test _x_complexf ≈ x
                end
            end

            # BFFT
            x_scaled = prod(size(x, d) for d in dims) .* x 
            @test bfft(x_fft, dims) ≈ x_scaled
            if test_inplace
                _x_fft = copy(x_fft)
                @test bfft!(_x_fft, dims) ≈ x_scaled
                @test _x_fft ≈ x_scaled
            end
            # test OOP plans. Just 1 plan to test, but we use a for loop for consistent style
            for P in (plan_bfft(similar(x_fft), dims),)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                @test P * x_fft ≈ x_scaled
                @test P \ (P * x_fft) ≈ x_fft
                _x_complexf = similar(x_complexf)
                @test mul!(_x_complexf, P, x_fft) ≈ x_scaled
                @test _x_complexf ≈ x_scaled
                if test_adjoint
                    _adjoint_test(P, x_complexf)
                end
            end
            # test IIP plans
            for P in (plan_bfft!(similar(x_fft), dims),)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                _x_fft = copy(x_fft)
                @test P * _x_fft ≈ x_scaled 
                @test _x_fft ≈ x_scaled 
                @test P \ _x_fft ≈ x_fft
                @test _x_fft ≈ x_fft
            end

            # IFFT
            @test ifft(x_fft, dims) ≈ x
            if test_inplace
                _x_fft = copy(x_fft)
                @test ifft!(_x_fft, dims) ≈ x
                @test _x_fft ≈ x
            end
            # test OOP plans
            for P in (plan_ifft(similar(x_complexf), dims), inv(plan_fft(similar(x_complexf), dims)))
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                @test P * x_fft ≈ x
                @test P \ (P * x_fft) ≈ x_fft
                _x_complexf = similar(x_complexf)
                @test mul!(_x_complexf, P, x_fft) ≈ x
                @test _x_complexf ≈ x
                if test_adjoint
                    _adjoint_test(P, x_complexf)
                end
            end
            # test IIP plans
            if test_inplace
                for P in (plan_ifft!(similar(x_complexf), dims), inv(plan_fft!(similar(x_complexf), dims)))
                    @test eltype(P) <: Complex
                    @test fftdims(P) == dims
                    _x_fft = copy(x_fft)
                    @test P * _x_fft ≈ x
                    @test _x_fft ≈ x
                    @test P \ _x_fft ≈ x_fft
                    @test _x_fft ≈ x_fft
                end
            end
        end
    end
end

"""
    TestUtils.test_real_fft(ArrayType=Array; test_real=true, test_inplace=true)

Run tests to verify correctness of RFFT/BRFFT/IRFFT functionality using a particular backend plan implementation. 
The backend implementation is assumed to be loaded prior to calling this function.

# Arguments

- `ArrayType`: determines the `AbstractArray` implementation for
  which the correctness tests are run. Arrays are constructed via
  `convert(ArrayType, ...)`.
- `test_inplace=true`: whether to test in-place plans. 
- `test_adjoint=true`: whether to test adjoints of plans. 
"""
function TestUtils.test_real_fft(ArrayType=Array; test_inplace=true, test_adjoint=true)
    @testset "correctness of rfft, brfft, irfft" begin
        for test_case in TEST_CASES
            _x, dims, _x_fft = test_case.x, test_case.dims, test_case.x_fft
            x = convert(ArrayType, _x) # dummy array that will be passed to plans
            x_real = float.(x) # for testing mutating real FFTs
            x_fft = convert(ArrayType, _x_fft)
            x_rfft = selectdim(x_fft, first(dims), 1:(size(x_fft, first(dims)) ÷ 2 + 1))

            if !(eltype(x) <: Real)
                continue
            end

            # RFFT
            @test rfft(x, dims) ≈ x_rfft
            for P in (plan_rfft(similar(x_real), dims), inv(plan_irfft(similar(x_rfft), size(x, first(dims)), dims)))
                @test eltype(P) <: Real
                @test fftdims(P) == dims
                # Always copy input before application due to FFTW real plans possibly mutating input (AbstractFFTs.jl#101)
                @test P * copy(x) ≈ x_rfft
                @test P \ (P * copy(x)) ≈ x
                _x_rfft = similar(x_rfft)
                @test mul!(_x_rfft, P, copy(x_real)) ≈ x_rfft
                @test _x_rfft ≈ x_rfft
                if test_adjoint
                    _adjoint_test(P, x_real; real_plan=true)
                end
            end

            # BRFFT
            x_scaled = prod(size(x, d) for d in dims) .* x
            @test brfft(x_rfft, size(x, first(dims)), dims) ≈ x_scaled
            for P in (plan_brfft(similar(x_rfft), size(x, first(dims)), dims),)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                @test P * copy(x_rfft) ≈ x_scaled 
                @test P \ (P * copy(x_rfft)) ≈ x_rfft
                _x_scaled = similar(x_real)
                @test mul!(_x_scaled, P, copy(x_rfft)) ≈ x_scaled 
                @test _x_scaled ≈ x_scaled
            end

            # IRFFT
            @test irfft(x_rfft, size(x, first(dims)), dims) ≈ x
            for P in (plan_irfft(similar(x_rfft), size(x, first(dims)), dims), inv(plan_rfft(similar(x_real), dims)))
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                @test P * copy(x_rfft) ≈ x 
                @test P \ (P * copy(x_rfft)) ≈ x_rfft
                _x_real = similar(x_real)
                @test mul!(_x_real, P, copy(x_rfft)) ≈ x_real
            end
        end
    end
end

end