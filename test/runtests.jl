using Random
using Test
using AbstractFFTs
using ChainRulesTestUtils
import Unitful
using LinearAlgebra
using ChainRulesCore
using FiniteDifferences

Random.seed!(1234)

# Load example plan implementation.
include("TestPlans.jl")

# Run interface tests for TestPlans 
AbstractFFTs.TestUtils.test_complex_ffts(Array)
AbstractFFTs.TestUtils.test_real_ffts(Array)

@testset "rfft sizes" begin
    A = rand(11, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 1)) == (6, 10)
    @test @inferred(AbstractFFTs.rfft_output_size(A, 2)) == (11, 6)
    A1 = rand(6, 10); A2 = rand(11, 6)
    @test @inferred(AbstractFFTs.brfft_output_size(A1, 11, 1)) == (11, 10)
    @test @inferred(AbstractFFTs.brfft_output_size(A2, 10, 2)) == (11, 10)
    @test_throws AssertionError AbstractFFTs.brfft_output_size(A1, 10, 2)
end

@testset "Shift functions" begin
    @test @inferred(AbstractFFTs.fftshift([1 2 3])) == [3 1 2]
    @test @inferred(AbstractFFTs.fftshift([1, 2, 3])) == [3, 1, 2]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6])) == [6 4 5; 3 1 2]
    a = [0 0 0]
    b = [0, 0, 0]
    c = [0 0 0; 0 0 0]
    @test (AbstractFFTs.fftshift!(a, [1 2 3]); a == [3 1 2])
    @test (AbstractFFTs.fftshift!(b, [1, 2, 3]); b == [3, 1, 2])
    @test (AbstractFFTs.fftshift!(c, [1 2 3; 4 5 6]); c == [6 4 5; 3 1 2])

    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], 1)) == [4 5 6; 1 2 3]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], ())) == [1 2 3; 4 5 6]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], (1,2))) == [6 4 5; 3 1 2]
    @test @inferred(AbstractFFTs.fftshift([1 2 3; 4 5 6], 1:2)) == [6 4 5; 3 1 2]
    @test (AbstractFFTs.fftshift!(c, [1 2 3; 4 5 6], 1); c == [4 5 6; 1 2 3])
    @test (AbstractFFTs.fftshift!(c, [1 2 3; 4 5 6], ()); c == [1 2 3; 4 5 6])
    @test (AbstractFFTs.fftshift!(c, [1 2 3; 4 5 6], (1,2)); c == [6 4 5; 3 1 2])
    @test (AbstractFFTs.fftshift!(c, [1 2 3; 4 5 6], 1:2); c == [6 4 5; 3 1 2])

    @test @inferred(AbstractFFTs.ifftshift([1 2 3])) == [2 3 1]
    @test @inferred(AbstractFFTs.ifftshift([1, 2, 3])) == [2, 3, 1]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6])) == [5 6 4; 2 3 1]
    @test (AbstractFFTs.ifftshift!(a, [1 2 3]); a == [2 3 1])
    @test (AbstractFFTs.ifftshift!(b, [1, 2, 3]); b == [2, 3, 1])
    @test (AbstractFFTs.ifftshift!(c, [1 2 3; 4 5 6]); c == [5 6 4; 2 3 1])

    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1)) == [4 5 6; 1 2 3]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], ())) == [1 2 3; 4 5 6]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], (1,2))) == [5 6 4; 2 3 1]
    @test @inferred(AbstractFFTs.ifftshift([1 2 3; 4 5 6], 1:2)) == [5 6 4; 2 3 1]
    @test (AbstractFFTs.ifftshift!(c, [1 2 3; 4 5 6], 1); c == [4 5 6; 1 2 3])
    @test (AbstractFFTs.ifftshift!(c, [1 2 3; 4 5 6], ()); c == [1 2 3; 4 5 6])
    @test (AbstractFFTs.ifftshift!(c, [1 2 3; 4 5 6], (1,2)); c == [5 6 4; 2 3 1])
    @test (AbstractFFTs.ifftshift!(c, [1 2 3; 4 5 6], 1:2); c == [5 6 4; 2 3 1])
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
    f9(p::AbstractFFTs.Plan{T}, sz) where {T} = AbstractFFTs.normalization(real(T), sz, fftdims(p))
    @test @inferred(f9(plan_fft(zeros(10), 1), 10)) == 1/10
end

@testset "output size" begin
    @testset "complex fft output size" begin
        for x_shape in ((3,), (3, 4), (3, 4, 5))
            N = length(x_shape)
            real_x = randn(x_shape)
            complex_x = randn(ComplexF64, x_shape)
            for x in (real_x, complex_x)
                for dims in unique((1, 1:N, N))
                    P = plan_fft(x, dims)
                    @test @inferred(AbstractFFTs.output_size(P)) == size(x)
                    @test AbstractFFTs.output_size(P') == size(x)
                    Pinv = plan_ifft(x)
                    @test AbstractFFTs.output_size(Pinv) == size(x)
                    @test AbstractFFTs.output_size(Pinv') == size(x)
                end
            end
        end
    end
    @testset "real fft output size" begin
        for x in (randn(3), randn(4), randn(3, 4), randn(3, 4, 5)) # test odd and even lengths
            N = ndims(x)
            for dims in unique((1, 1:N, N))
                P = plan_rfft(x, dims)        
                Px_sz = size(P * x)
                @test AbstractFFTs.output_size(P) == Px_sz 
                @test AbstractFFTs.output_size(P') == size(x) 
                y = randn(ComplexF64, Px_sz)
                Pinv = plan_irfft(y, size(x)[first(dims)], dims)
                @test AbstractFFTs.output_size(Pinv) == size(Pinv * y)
                @test AbstractFFTs.output_size(Pinv') == size(y)
            end
        end
    end
end

# Test that dims defaults to 1:ndims for fft-like functions
@testset "Default dims" begin
    for x in (randn(3), randn(3, 4), randn(3, 4, 5))
        N = ndims(x)
        complex_x = complex.(x)
        @test fft(x) ≈ fft(x, 1:N)
        @test ifft(x) ≈ ifft(x, 1:N)
        @test bfft(x) ≈ bfft(x, 1:N)
        @test rfft(x) ≈ rfft(x, 1:N)
        d = 2 * size(x, 1) - 1
        @test irfft(x, d) ≈ irfft(x, d, 1:N)
        @test brfft(x, d) ≈ brfft(x, d, 1:N)
    end
end

@testset "Complex float promotion" begin
    for x in (rand(-5:5, 3), rand(-5:5, 3, 4), rand(-5:5, 3, 4, 5))
        N = ndims(x)
        @test fft(x) ≈ fft(complex.(x)) ≈ fft(complex.(float.(x)))
        @test ifft(x) ≈ ifft(complex.(x)) ≈ ifft(complex.(float.(x)))
        @test bfft(x) ≈ bfft(complex.(x)) ≈ bfft(complex.(float.(x)))
        d = 2 * size(x, 1) - 1
        @test irfft(x, d) ≈ irfft(complex.(x), d) ≈ irfft(complex.(float.(x)), d)
        @test brfft(x, d) ≈ brfft(complex.(x), d) ≈ brfft(complex.(float.(x)), d)
    end
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
        # Overloads to allow ChainRulesTestUtils to test rules w.r.t. ScaledPlan's. See https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/256
        InnerPlan = Union{TestPlans.TestPlan, TestPlans.InverseTestPlan, TestPlans.TestRPlan, TestPlans.InverseTestRPlan}
        function FiniteDifferences.to_vec(x::InnerPlan)
            function FFTPlan_from_vec(x_vec::Vector)
                return x
            end
            return Bool[], FFTPlan_from_vec
        end
        ChainRulesTestUtils.test_approx(::ChainRulesCore.AbstractZero, x::InnerPlan, msg=""; kwargs...) = true
        ChainRulesTestUtils.rand_tangent(::AbstractRNG, x::InnerPlan) = ChainRulesCore.NoTangent()

        for x_shape in ((2,), (2, 3), (3, 4, 5))
            N = length(x_shape)
            x = randn(x_shape)
            complex_x = randn(ComplexF64, x_shape)
            Δ = (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesTestUtils.rand_tangent(complex_x))
            for dims in unique((1, 1:N, N))
                # fft, ifft, bfft
                for f in (fft, ifft, bfft)
                    test_frule(f, x, dims)
                    test_rrule(f, x, dims)
                    test_frule(f, complex_x, dims)
                    test_rrule(f, complex_x, dims)
                end
                for (pf, pf!) in ((plan_fft, plan_fft!), (plan_ifft, plan_ifft!), (plan_bfft, plan_bfft!)) 
                    test_frule(*, pf(x, dims), x)
                    test_rrule(*, pf(x, dims), x)
                    test_frule(*, pf(complex_x, dims), complex_x)
                    test_rrule(*, pf(complex_x, dims), complex_x)

                    @test_throws ArgumentError ChainRulesCore.frule(Δ, *, pf!(complex_x, dims), complex_x)
                    @test_throws ArgumentError ChainRulesCore.rrule(*, pf!(complex_x, dims), complex_x)
                end

                # rfft 
                test_frule(rfft, x, dims)
                test_rrule(rfft, x, dims)
                test_frule(*, plan_rfft(x, dims), x)
                test_rrule(*, plan_rfft(x, dims), x)

                # irfft, brfft
                for f in (irfft, brfft)
                    for d in (2 * size(x, first(dims)) - 1, 2 * size(x, first(dims)) - 2)
                        test_frule(f, x, d, dims)
                        test_rrule(f, x, d, dims)
                        test_frule(f, complex_x, d, dims)
                        test_rrule(f, complex_x, d, dims)
                    end
                end
                for pf in (plan_irfft, plan_brfft)
                    for d in (2 * size(x, first(dims)) - 1, 2 * size(x, first(dims)) - 2)
                        test_frule(*, pf(complex_x, d, dims), complex_x)
                        test_rrule(*, pf(complex_x, d, dims), complex_x) 
                    end
                end
            end
        end
    end
end
            
