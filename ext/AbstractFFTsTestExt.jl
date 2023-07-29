# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

module AbstractFFTsTestExt

using AbstractFFTs
using AbstractFFTs: TestUtils
using AbstractFFTs.LinearAlgebra
using Test

# Ground truth x_fft computed using FFTW library
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


function TestUtils.test_plan(P::AbstractFFTs.Plan, x::AbstractArray, x_transformed::AbstractArray; inplace_plan=false, copy_input=false)
    _copy = copy_input ? copy : identity
    if !inplace_plan
        @test P * _copy(x) ≈ x_transformed
        @test P \ (P * _copy(x)) ≈ x
        _x_out = similar(P * _copy(x))
        @test mul!(_x_out, P, _copy(x)) ≈ x_transformed
        @test _x_out ≈ x_transformed
    else
        _x = copy(x)
        @test P * _copy(_x) ≈ x_transformed
        @test _x ≈ x_transformed
        @test P \ _copy(_x) ≈ x
        @test _x ≈ x
    end
end

function TestUtils.test_plan_adjoint(P::AbstractFFTs.Plan, x::AbstractArray; real_plan=false, copy_input=false)
    _copy = copy_input ? copy : identity
    y = rand(eltype(P * _copy(x)), size(P * _copy(x)))
    # test basic properties
    @test_skip eltype(P') === typeof(y) # (AbstractFFTs.jl#110)
    @test (P')' === P # test adjoint of adjoint
    @test size(P') == AbstractFFTs.output_size(P) # test size of adjoint 
    # test correctness of adjoint and its inverse via the dot test
    if !real_plan
        @test dot(y, P * _copy(x)) ≈ dot(P' * _copy(y), x)
        @test dot(y, P \ _copy(x)) ≈ dot(P' \ _copy(y), x) 
    else
        _component_dot(x, y) = dot(real.(x), real.(y)) + dot(imag.(x), imag.(y))
        @test _component_dot(y, P * _copy(x)) ≈ _component_dot(P' * _copy(y), x)
        @test _component_dot(x, P \ _copy(y)) ≈ _component_dot(P' \ _copy(x), y) 
    end
    @test_throws MethodError mul!(x, P', y)
end

function TestUtils.test_complex_ffts(ArrayType=Array; test_inplace=true, test_adjoint=true) 
    @testset "correctness of fft, bfft, ifft" begin
        for test_case in TEST_CASES
            _x, dims, _x_fft = copy(test_case.x), test_case.dims, copy(test_case.x_fft)
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
            # test OOP plans, checking plan_fft and also inv and plan_inv of plan_ifft, 
            # which should give functionally identical plans
            for P in (plan_fft(similar(x_complexf), dims), 
                      (_inv(plan_ifft(similar(x_complexf), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_complexf, x_fft)
                if test_adjoint
                    @test fftdims(P') == fftdims(P)
                    TestUtils.test_plan_adjoint(P, x_complexf)
                end
            end
            if test_inplace
                # test IIP plans
                for P in (plan_fft!(similar(x_complexf), dims), 
                          (_inv(plan_ifft!(similar(x_complexf), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                    TestUtils.test_plan(P, x_complexf, x_fft; inplace_plan=true)
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
                TestUtils.test_plan(P, x_fft, x_scaled)
                if test_adjoint
                    TestUtils.test_plan_adjoint(P, x_fft)
                end
            end
            # test IIP plans
            for P in (plan_bfft!(similar(x_fft), dims),)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_fft, x_scaled; inplace_plan=true)
            end

            # IFFT
            @test ifft(x_fft, dims) ≈ x
            if test_inplace
                _x_fft = copy(x_fft)
                @test ifft!(_x_fft, dims) ≈ x
                @test _x_fft ≈ x
            end
            # test OOP plans
            for P in (plan_ifft(similar(x_complexf), dims), 
                      (_inv(plan_fft(similar(x_complexf), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_fft, x)
                if test_adjoint
                    TestUtils.test_plan_adjoint(P, x_fft)
                end
            end
            # test IIP plans
            if test_inplace
                for P in (plan_ifft!(similar(x_complexf), dims), 
                          (_inv(plan_fft!(similar(x_complexf), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                    @test eltype(P) <: Complex
                    @test fftdims(P) == dims
                    TestUtils.test_plan(P, x_fft, x; inplace_plan=true)
                end
            end
        end
    end
end

function TestUtils.test_real_ffts(ArrayType=Array; test_adjoint=true, copy_input=false)
    @testset "correctness of rfft, brfft, irfft" begin
        for test_case in TEST_CASES
            _x, dims, _x_fft = copy(test_case.x), test_case.dims, copy(test_case.x_fft)
            x = convert(ArrayType, _x) # dummy array that will be passed to plans
            x_real = float.(x) # for testing mutating real FFTs
            x_fft = convert(ArrayType, _x_fft)
            x_rfft = collect(selectdim(x_fft, first(dims), 1:(size(x_fft, first(dims)) ÷ 2 + 1)))

            if !(eltype(x) <: Real)
                continue
            end

            # RFFT
            @test rfft(x, dims) ≈ x_rfft
            for P in (plan_rfft(similar(x_real), dims), 
                      (_inv(plan_irfft(similar(x_rfft), size(x, first(dims)), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                @test eltype(P) <: Real
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_real, x_rfft; copy_input=copy_input)
                if test_adjoint
                    TestUtils.test_plan_adjoint(P, x_real; real_plan=true, copy_input=copy_input)
                end
            end

            # BRFFT
            x_scaled = prod(size(x, d) for d in dims) .* x
            @test brfft(x_rfft, size(x, first(dims)), dims) ≈ x_scaled
            for P in (plan_brfft(similar(x_rfft), size(x, first(dims)), dims),)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_rfft, x_scaled; copy_input=copy_input)
                if test_adjoint
                    TestUtils.test_plan_adjoint(P, x_rfft; real_plan=true, copy_input=copy_input)
                end
            end

            # IRFFT
            @test irfft(x_rfft, size(x, first(dims)), dims) ≈ x
            for P in (plan_irfft(similar(x_rfft), size(x, first(dims)), dims), 
                      (_inv(plan_rfft(similar(x_real), dims)) for _inv in (inv, AbstractFFTs.plan_inv))...)
                @test eltype(P) <: Complex
                @test fftdims(P) == dims
                TestUtils.test_plan(P, x_rfft, x; copy_input=copy_input)
                if test_adjoint
                    TestUtils.test_plan_adjoint(P, x_rfft; real_plan=true, copy_input=copy_input)
                end
            end
        end
    end
end

end
