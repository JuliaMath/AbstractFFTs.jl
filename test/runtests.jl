# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

using AbstractFFTs
using AbstractFFTs: Plan
using LinearAlgebra
using Test

@testset "rfft sizes" begin
    A = rand(11, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 1)) == (6, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 2)) == (11, 6)
    A1 = rand(6, 10); A2 = rand(11, 6)
    @test @inferred(AbstractFFTs.brfft_output_size(A1, 11, 1)) == (11, 10)
    @test @inferred(AbstractFFTs.brfft_output_size(A2, 10, 2)) == (11, 10)
    @test_throws AssertionError AbstractFFTs.brfft_output_size(A1, 10, 2)
end

mutable struct TestPlan{T} <: Plan{T}
    region
    pinv::Plan{T}
    TestPlan{T}(region) where {T} = new{T}(region)
end

mutable struct InverseTestPlan{T} <: Plan{T}
    region
    pinv::Plan{T}
    InverseTestPlan{T}(region) where {T} = new{T}(region)
end

AbstractFFTs.plan_fft(x::Vector{T}, region; kwargs...) where {T} = TestPlan{T}(region)
AbstractFFTs.plan_bfft(x::Vector{T}, region; kwargs...) where {T} = InverseTestPlan{T}(region)
AbstractFFTs.plan_inv(p::TestPlan{T}) where {T} = InverseTestPlan{T}

# Just a helper function since forward and backward are nearly identical
function dft!(y::Vector, x::Vector, sign::Int)
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    fill!(y, zero(complex(float(eltype(x)))))
    c = sign * 2π / n
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(c*j*k)
    end
    return y
end

mul!(y::Vector, p::TestPlan, x::Vector) = dft!(y, x, -1)
mul!(y::Vector, p::InverseTestPlan, x::Vector) = dft!(y, x, 1)

Base.:*(p::TestPlan, x::Vector) = mul!(copy(x), p, x)
Base.:*(p::InverseTestPlan, x::Vector) = mul!(copy(x), p, x)

@testset "Custom Plan" begin
    x = AbstractFFTs.fft(collect(1:8))
    # Result computed using FFTW
    fftw_fft = [36.0 + 0.0im,
                -4.0 + 9.65685424949238im,
                -4.0 + 4.0im,
                -4.0 + 1.6568542494923806im,
                -4.0 + 0.0im,
                -4.0 - 1.6568542494923806im,
                -4.0 - 4.0im,
                -4.0 - 9.65685424949238im]
    @test x ≈ fftw_fft

    fftw_bfft = [Complex{Float64}(8i, 0) for i in 1:8]
    @test AbstractFFTs.bfft(x) ≈ fftw_bfft

    fftw_ifft = [Complex{Float64}(i, 0) for i in 1:8]
    @test AbstractFFTs.ifft(x) ≈ fftw_ifft

    @test eltype(plan_fft(collect(1:8))) == Int
end

@testset "Shift functions" begin
    @test AbstractFFTs.fftshift([1 2 3]) == [3 1 2]
    @test AbstractFFTs.fftshift([1, 2, 3]) == [3, 1, 2]
    @test AbstractFFTs.fftshift([1 2 3; 4 5 6]) == [6 4 5; 3 1 2]

    @test AbstractFFTs.fftshift([1 2 3; 4 5 6], 1) == [4 5 6; 1 2 3]
    @test AbstractFFTs.fftshift([1 2 3; 4 5 6], ()) == [1 2 3; 4 5 6]
    @test AbstractFFTs.fftshift([1 2 3; 4 5 6], (1,2)) == [6 4 5; 3 1 2]
    @test AbstractFFTs.fftshift([1 2 3; 4 5 6], 1:2) == [6 4 5; 3 1 2]

    @test AbstractFFTs.ifftshift([1 2 3]) == [2 3 1]
    @test AbstractFFTs.ifftshift([1, 2, 3]) == [2, 3, 1]
    @test AbstractFFTs.ifftshift([1 2 3; 4 5 6]) == [5 6 4; 2 3 1]

    @test AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1) == [4 5 6; 1 2 3]
    @test AbstractFFTs.ifftshift([1 2 3; 4 5 6], ()) == [1 2 3; 4 5 6]
    @test AbstractFFTs.ifftshift([1 2 3; 4 5 6], (1,2)) == [5 6 4; 2 3 1]
    @test AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1:2) == [5 6 4; 2 3 1]
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
        for f in (fftfreq, rfftfreq), n in (8, 9), multiplier in (2, 1/3, -1/7)
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
