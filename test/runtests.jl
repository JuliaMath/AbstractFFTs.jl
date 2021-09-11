# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

using AbstractFFTs
using AbstractFFTs: Plan
using ChainRulesTestUtils

using LinearAlgebra
using Random
using Test

import Unitful

Random.seed!(1234)

include("testplans.jl")

@testset "rfft sizes" begin
    A = rand(11, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 1)) == (6, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 2)) == (11, 6)
    A1 = rand(6, 10); A2 = rand(11, 6)
    @test @inferred(AbstractFFTs.brfft_output_size(A1, 11, 1)) == (11, 10)
    @test @inferred(AbstractFFTs.brfft_output_size(A2, 10, 2)) == (11, 10)
    @test_throws AssertionError AbstractFFTs.brfft_output_size(A1, 10, 2)
end

@testset "Custom Plan" begin
    # DFT along last dimension, results computed using FFTW
    for (x, fftw_fft) in (
        (collect(1:7),
         [28.0 + 0.0im,
          -3.5 + 7.267824888003178im,
          -3.5 + 2.7911568610884143im,
          -3.5 + 0.7988521603655248im,
          -3.5 - 0.7988521603655248im,
          -3.5 - 2.7911568610884143im,
          -3.5 - 7.267824888003178im]),
        (collect(1:8),
         [36.0 + 0.0im,
          -4.0 + 9.65685424949238im,
          -4.0 + 4.0im,
          -4.0 + 1.6568542494923806im,
          -4.0 + 0.0im,
          -4.0 - 1.6568542494923806im,
          -4.0 - 4.0im,
          -4.0 - 9.65685424949238im]),
        (collect(reshape(1:8, 2, 4)),
         [16.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im;
          20.0+0.0im  -4.0+4.0im  -4.0+0.0im  -4.0-4.0im]),
        (collect(reshape(1:9, 3, 3)),
         [12.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
          15.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im;
          18.0+0.0im  -4.5+2.598076211353316im  -4.5-2.598076211353316im]),
    )
        # FFT
        dims = ndims(x)
        y = AbstractFFTs.fft(x, dims)
        @test y ≈ fftw_fft
        P = plan_fft(x, dims)
        @test eltype(P) === ComplexF64
        @test P * x ≈ fftw_fft
        @test P \ (P * x) ≈ x

        fftw_bfft = complex.(size(x, dims) .* x)
        @test AbstractFFTs.bfft(y, dims) ≈ fftw_bfft
        P = plan_bfft(x, dims)
        @test P * y ≈ fftw_bfft
        @test P \ (P * y) ≈ y

        fftw_ifft = complex.(x)
        @test AbstractFFTs.ifft(y, dims) ≈ fftw_ifft
        P = plan_ifft(x, dims)
        @test P * y ≈ fftw_ifft
        @test P \ (P * y) ≈ y

        # real FFT
        fftw_rfft = fftw_fft[
            (Colon() for _ in 1:(ndims(fftw_fft) - 1))...,
            1:(size(fftw_fft, ndims(fftw_fft)) ÷ 2 + 1)
        ]
        ry = AbstractFFTs.rfft(x, dims)
        @test ry ≈ fftw_rfft
        P = plan_rfft(x, dims)
        @test eltype(P) === Int
        @test P * x ≈ fftw_rfft
        @test P \ (P * x) ≈ x

        fftw_brfft = complex.(size(x, dims) .* x)
        @test AbstractFFTs.brfft(ry, size(x, dims), dims) ≈ fftw_brfft
        P = plan_brfft(ry, size(x, dims), dims)
        @test P * ry ≈ fftw_brfft
        @test P \ (P * ry) ≈ ry
        
        fftw_irfft = complex.(x)
        @test AbstractFFTs.irfft(ry, size(x, dims), dims) ≈ fftw_irfft
        P = plan_irfft(ry, size(x, dims), dims)
        @test P * ry ≈ fftw_irfft
        @test P \ (P * ry) ≈ ry
    end
end

@testset "Shift functions" begin
    @test @inferred(AbstractFFTs.fftshift([1 2 3])) == [3 1 2]
    @test @inferred(AbstractFFTs.fftshift([1, 2, 3])) == [3, 1, 2]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6])) == [6 4 5; 3 1 2]

    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], 1)) == [4 5 6; 1 2 3]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], ())) == [1 2 3; 4 5 6]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], (1,2))) == [6 4 5; 3 1 2]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], 1:2)) == [6 4 5; 3 1 2]

    @test @inferred(AbstractFFTs.ifftshift([1 2 3])) == [2 3 1]
    @test @inferred(AbstractFFTs.ifftshift([1, 2, 3])) == [2, 3, 1]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6])) == [5 6 4; 2 3 1]

    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1)) == [4 5 6; 1 2 3]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], ())) == [1 2 3; 4 5 6]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], (1,2))) == [5 6 4; 2 3 1]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1:2)) == [5 6 4; 2 3 1]
end

@testset "FFT Frequencies" begin
    @test fftfreq(8) isa Frequencies
    @test copy(fftfreq(8)) isa Frequencies

    # N even
    @test fftfreq(8) == [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
    @test rfftfreq(8) == [0.0, 0.125, 0.25, 0.375, 0.5]
    @test fftshift(fftfreq(8)) == -0.5:0.125:0.375

    # N odd
    @test fftfreq(5) == [0.0, 0.2, 0.4, -0.4, -0.2]
    @test rfftfreq(5) == [0.0, 0.2, 0.4]
    @test fftshift(fftfreq(5)) == -0.4:0.2:0.4

    # Sampling Frequency
    @test fftfreq(5, 2) == [0.0, 0.4, 0.8, -0.8, -0.4]
    # <:Number type compatibility
    @test eltype(fftfreq(5, ComplexF64(2))) == ComplexF64

    @test_throws ArgumentError Frequencies(12, 10, 1)

    @testset "scaling" begin
        @test fftfreq(4, 1) * 2 === fftfreq(4, 2)
        @test fftfreq(4, 1) .* 2 === fftfreq(4, 2)
        @test 2 * fftfreq(4, 1) === fftfreq(4, 2)
        @test 2 .* fftfreq(4, 1) === fftfreq(4, 2)

        @test fftfreq(4, 1) / 2 === fftfreq(4, 1/2)
        @test fftfreq(4, 1) ./ 2 === fftfreq(4, 1/2)

        @test 2 \ fftfreq(4, 1) === fftfreq(4, 1/2)
        @test 2 .\ fftfreq(4, 1) === fftfreq(4, 1/2)
    end

    @testset "extrema" begin
        function check_extrema(freqs)
            for f in [minimum, maximum, extrema]
                @test f(freqs) == f(collect(freqs)) == f(fftshift(freqs))
            end
        end
        for f in (fftfreq, rfftfreq), n in (8, 9), multiplier in (2, 1/3, -1/7, 1.0*Unitful.mm)
            freqs = f(n, multiplier)
            check_extrema(freqs)
        end
    end
end

@testset "normalization" begin
    # normalization should be inferable even if region is only inferred as ::Any,
    # need to wrap in another function to test this (note that p.region::Any for
    # p::TestPlan)
    f9(p::Plan{T}, sz) where {T} = AbstractFFTs.normalization(real(T), sz, p.region)
    @test @inferred(f9(plan_fft(zeros(10), 1), 10)) == 1/10
end

@testset "ChainRules" begin
    @testset "shift functions" begin
        for x in (randn(3), randn(3, 4), randn(3, 4, 5))
            for dims in ((), 1, 2, (1,2), 1:2)
                any(d > ndims(x) for d in dims) && continue

                # type inference checks of `rrule` fail on old Julia versions
                # for higher-dimensional arrays:
                # https://github.com/JuliaMath/AbstractFFTs.jl/pull/58#issuecomment-916530016
                check_inferred = ndims(x) < 3 || VERSION >= v"1.6"

                test_frule(AbstractFFTs.fftshift, x, dims)
                test_rrule(AbstractFFTs.fftshift, x, dims; check_inferred=check_inferred)

                test_frule(AbstractFFTs.ifftshift, x, dims)
                test_rrule(AbstractFFTs.ifftshift, x, dims; check_inferred=check_inferred)
            end
        end
    end

    @testset "fft" begin
        for x in (randn(3), randn(3, 4), randn(3, 4, 5))
            N = ndims(x)
            complex_x = complex.(x)
            for dims in unique((1, 1:N, N))
                for f in (fft, ifft, bfft)
                    test_frule(f, x, dims)
                    test_rrule(f, x, dims)
                    test_frule(f, complex_x, dims)
                    test_rrule(f, complex_x, dims)
                end

                test_frule(rfft, x, dims)
                test_rrule(rfft, x, dims)

                for f in (irfft, brfft)
                    for d in (2 * size(x, first(dims)) - 1, 2 * size(x, first(dims)) - 2)
                        test_frule(f, x, d, dims)
                        test_rrule(f, x, d, dims)
                        test_frule(f, complex_x, d, dims)
                        test_rrule(f, complex_x, d, dims)
                    end
                end
            end
        end
    end
end
