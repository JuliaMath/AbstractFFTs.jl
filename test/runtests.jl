# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

using AbstractFFTs
using Base.Test

import AbstractFFTs: Plan, plan_fft, plan_inv, plan_bfft
import Base: A_mul_B!, *

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

Base.A_mul_B!(y::Vector, p::TestPlan, x::Vector) = dft!(y, x, -1)
Base.A_mul_B!(y::Vector, p::InverseTestPlan, x::Vector) = dft!(y, x, 1)

Base.:*(p::TestPlan, x::Vector) = A_mul_B!(copy(x), p, x)
Base.:*(p::InverseTestPlan, x::Vector) = A_mul_B!(copy(x), p, x)

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
