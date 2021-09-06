# This file contains code that was formerly part of Julia. License is MIT: https://julialang.org/license

using AbstractFFTs
using AbstractFFTs: Plan
using ChainRulesTestUtils

using LinearAlgebra
using Random
using Test

import Unitful

Random.seed!(1234)

@testset "rfft sizes" begin
    A = rand(11, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 1)) == (6, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 2)) == (11, 6)
    A1 = rand(6, 10); A2 = rand(11, 6)
    @test @inferred(AbstractFFTs.brfft_output_size(A1, 11, 1)) == (11, 10)
    @test @inferred(AbstractFFTs.brfft_output_size(A2, 10, 2)) == (11, 10)
    @test_throws AssertionError AbstractFFTs.brfft_output_size(A1, 10, 2)
end

mutable struct TestPlan{T,N} <: Plan{T}
    region
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function TestPlan{T}(region, sz::NTuple{N,Int}) where {T,N}
        return new{T,N}(region, sz)
    end
end

mutable struct InverseTestPlan{T,N} <: Plan{T}
    region
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function InverseTestPlan{T}(region, sz::NTuple{N,Int}) where {T,N}
        return new{T,N}(region, sz)
    end
end

Base.size(p::TestPlan) = p.sz
Base.ndims(::TestPlan{T,N}) where {T,N} = N
Base.size(p::InverseTestPlan) = p.sz
Base.ndims(::InverseTestPlan{T,N}) where {T,N} = N

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...) where {T}
    return TestPlan{T}(region, size(x))
end
function AbstractFFTs.plan_bfft(x::AbstractArray{T}, region; kwargs...) where {T}
    return InverseTestPlan{T}(region, size(x))
end
function AbstractFFTs.plan_inv(p::TestPlan{T}) where {T}
    unscaled_pinv = InverseTestPlan{T}(p.region, p.sz)
    unscaled_pinv.pinv = p
    pinv = AbstractFFTs.ScaledPlan(
        unscaled_pinv, AbstractFFTs.normalization(T, p.sz, p.region),
    )
    return pinv
end
function AbstractFFTs.plan_inv(p::InverseTestPlan{T}) where {T}
    unscaled_pinv = TestPlan{T}(p.region, p.sz)
    unscaled_pinv.pinv = p
    pinv = AbstractFFTs.ScaledPlan(
        unscaled_pinv, AbstractFFTs.normalization(T, p.sz, p.region),
    )
    return pinv
end

# Just a helper function since forward and backward are nearly identical
# The function does not check if the size of `y` and `x` are compatible, this
# is done in the function where `dft!` is called since the check differs for FFTs
# with complex and real-valued signals
function dft!(
    y::AbstractArray{<:Complex,N},
    x::AbstractArray{<:Union{Complex,Real},N},
    dims,
    sign::Int
) where {N}
    # check that dimensions that are transformed are unique
    allunique(dims) || error("dimensions have to be unique")
    
    T = eltype(y)
    # we use `size(x, d)` since for real-valued signals
    # `size(y, first(dims)) = size(x, first(dims)) ÷ 2 + 1`
    cs = map(d -> T(sign * 2π / size(x, d)), dims)
    fill!(y, zero(T))
    for yidx in CartesianIndices(y)
        # set of indices of `x` on which `y[yidx]` depends
        xindices = CartesianIndices(
            ntuple(i -> i in dims ? axes(x, i) : yidx[i]:yidx[i], Val(N))
        )
        for xidx in xindices
            y[yidx] += x[xidx] * cis(sum(c * (yidx[d] - 1) * (xidx[d] - 1) for (c, d) in zip(cs, dims)))
        end
    end
    return y
end

function mul!(
    y::AbstractArray{<:Complex,N}, p::TestPlan, x::AbstractArray{<:Union{Complex,Real},N}
) where {N}
    size(y) == size(p) == size(x) || throw(DimensionMismatch())
    dft!(y, x, p.region, -1)
end
function mul!(
    y::AbstractArray{<:Complex,N}, p::InverseTestPlan, x::AbstractArray{<:Union{Complex,Real},N}
) where {N}
    size(y) == size(p) == size(x) || throw(DimensionMismatch())
    dft!(y, x, p.region, 1)
end

Base.:*(p::TestPlan, x::AbstractArray) = mul!(similar(x, complex(float(eltype(x)))), p, x)
Base.:*(p::InverseTestPlan, x::AbstractArray) = mul!(similar(x, complex(float(eltype(x)))), p, x)

mutable struct TestRPlan{T,N} <: Plan{T}
    region
    sz::NTuple{N,Int}
    pinv::Plan{T}
    TestRPlan{T}(region, sz::NTuple{N,Int}) where {T,N} = new{T,N}(region, sz)
end

mutable struct InverseTestRPlan{T,N} <: Plan{T}
    d::Int
    region
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function InverseTestRPlan{T}(d::Int, region, sz::NTuple{N,Int}) where {T,N}
        sz[first(region)::Int] == d ÷ 2 + 1 || error("incompatible dimensions")
        return new{T,N}(d, region, sz)
    end
end

function AbstractFFTs.plan_rfft(x::AbstractArray{T}, region; kwargs...) where {T}
    return TestRPlan{T}(region, size(x))
end
function AbstractFFTs.plan_brfft(x::AbstractArray{T}, d, region; kwargs...) where {T}
    return InverseTestRPlan{T}(d, region, size(x))
end
function AbstractFFTs.plan_inv(p::TestRPlan{T,N}) where {T,N}
    firstdim = first(p.region)::Int
    d = p.sz[firstdim]
    sz = ntuple(i -> i == firstdim ? d ÷ 2 + 1 : p.sz[i], Val(N))
    unscaled_pinv = InverseTestRPlan{T}(d, p.region, sz)
    unscaled_pinv.pinv = p
    pinv = AbstractFFTs.ScaledPlan(
        unscaled_pinv, AbstractFFTs.normalization(T, p.sz, p.region),
    )
    return pinv
end
function AbstractFFTs.plan_inv(p::InverseTestRPlan{T,N}) where {T,N}
    firstdim = first(p.region)::Int
    sz = ntuple(i -> i == firstdim ? p.d : p.sz[i], Val(N))
    unscaled_pinv = TestRPlan{T}(p.region, sz)
    unscaled_pinv.pinv = p
    pinv = AbstractFFTs.ScaledPlan(
        unscaled_pinv, AbstractFFTs.normalization(T, sz, p.region),
    )
    return pinv
end

Base.size(p::TestRPlan) = p.sz
Base.ndims(::TestRPlan{T,N}) where {T,N} = N
Base.size(p::InverseTestRPlan) = p.sz
Base.ndims(::InverseTestRPlan{T,N}) where {T,N} = N

function real_invdft!(
    y::AbstractArray{<:Real,N},
    x::AbstractArray{<:Union{Complex,Real},N},
    dims,
) where {N}
    # check that dimensions that are transformed are unique
    allunique(dims) || error("dimensions have to be unique")

    firstdim = first(dims)
    size_x_firstdim = size(x, firstdim)
    iseven_firstdim = iseven(size(y, firstdim))
    # we do not check that the input corresponds to a real-valued signal
    # (i.e., that the first and, if `iseven_firstdim`, the last value in dimension
    # `haldim` of `x` are real values) due to numerical inaccuracies
    # instead we just use the real part of these entries

    T = eltype(y)
    # we use `size(y, d)` since `size(x, first(dims)) = size(y, first(dims)) ÷ 2 + 1`
    cs = map(d -> T(2π / size(y, d)), dims)
    fill!(y, zero(T))
    for yidx in CartesianIndices(y)
        # set of indices of `x` on which `y[yidx]` depends
        xindices = CartesianIndices(
            ntuple(i -> i in dims ? axes(x, i) : yidx[i]:yidx[i], Val(N))
        )
        for xidx in xindices
            coeffimag, coeffreal = sincos(
                sum(c * (yidx[d] - 1) * (xidx[d] - 1) for (c, d) in zip(cs, dims))
            )

            # the first and, if `iseven_firstdim`, the last term of the DFT are scaled
            # with 1 instead of 2 and only the real part is used (see note above)
            xidx_firstdim = xidx[firstdim]
            if xidx_firstdim == 1 || (iseven_firstdim && xidx_firstdim == size_x_firstdim)
                y[yidx] += coeffreal * real(x[xidx])
            else
                xreal, ximag = reim(x[xidx])
                y[yidx] += 2 * (coeffreal * xreal - coeffimag * ximag)
            end
        end
    end

    return y
end

to_real!(x::AbstractArray) = map!(real, x, x)

function Base.:*(p::TestRPlan, x::AbstractArray)
    size(p) == size(x) || error("array and plan are not consistent")

    # create output array
    firstdim = first(p.region)::Int
    d = size(x, firstdim)
    firstdim_size = d ÷ 2 + 1
    T = complex(float(eltype(x)))
    sz = ntuple(i -> i == firstdim ? firstdim_size : size(x, i), Val(ndims(x)))
    y = similar(x, T, sz)

    # compute DFT
    dft!(y, x, p.region, -1)

    # we clean the output a bit to make sure that we return real values
    # whenever the output is mathematically guaranteed to be a real number
    to_real!(selectdim(y, firstdim, 1))
    if iseven(d)
        to_real!(selectdim(y, firstdim, firstdim_size))
    end

    return y
end

function Base.:*(p::InverseTestRPlan, x::AbstractArray)
    size(p) == size(x) || error("array and plan are not consistent")

    # create output array
    firstdim = first(p.region)::Int
    d = p.d
    sz = ntuple(i -> i == firstdim ? d : size(x, i), Val(ndims(x)))
    y = similar(x, real(float(eltype(x))), sz)

    # compute DFT
    real_invdft!(y, x, p.region)

    return y
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

                test_frule(AbstractFFTs.fftshift, x, dims)
                test_rrule(AbstractFFTs.fftshift, x, dims)

                test_frule(AbstractFFTs.fftshift, x, dims)
                test_rrule(AbstractFFTs.fftshift, x, dims)
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
