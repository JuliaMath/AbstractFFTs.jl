using AbstractFFTs
using ForwardDiff
using Test
using ForwardDiff: Dual, partials, value

# Needed until https://github.com/JuliaDiff/ForwardDiff.jl/pull/732 is merged
complexpartials(x, k) = partials(real(x), k) + im*partials(imag(x), k)

@testset "ForwardDiff extension tests" begin
    x1 = Dual.(1:4.0, 2:5, 3:6)
    c1 = Dual.(1:4.0, 2:5, 3:6) + im*Dual.(2:5.0, 3:6, 3:6)

    @test AbstractFFTs.complexfloat(x1)[1] === Dual(1.0, 2.0, 3.0) + 0im
    @test AbstractFFTs.realfloat(x1)[1] === Dual(1.0, 2.0, 3.0)

    @test fft(x1, 1)[1] isa Complex{<:Dual}
    @test plan_fft(x1, 1) * x1 == fft(x1, 1)
    @test size(plan_fft(x1,1)) == (4,)

    @testset "$f" for f in (fft, ifft, rfft, bfft)
        @test value.(f(x1)) == f(value.(x1))
        @test complexpartials.(f(x1), 1) == f(partials.(x1, 1))
        @test complexpartials.(f(x1), 2) == f(partials.(x1, 2))
    end

    @test ifft(fft(x1)) ≈ x1
    @test irfft(rfft(x1), length(x1)) ≈ x1
    @test brfft(rfft(x1), length(x1)) ≈ 4x1

    f = x -> real(fft([x; 0; 0])[1])
    @test ForwardDiff.derivative(f,0.1) ≈ 1

    r = x -> real(rfft([x; 0; 0])[1])
    @test ForwardDiff.derivative(r,0.1) ≈ 1


    n = 100
    θ = range(0,2π; length=n+1)[1:end-1]
    # emperical from Mathematical
    @test ForwardDiff.derivative(ω -> fft(exp.(ω .* cos.(θ)))[1]/n, 1) ≈ 0.565159103992485

    @testset "matrix" begin
        A = x1 * (1:10)'
        @test value.(fft(A)) == fft(value.(A))
        @test complexpartials.(fft(A), 1) == fft(partials.(A, 1))
        @test complexpartials.(fft(A), 2) == fft(partials.(A, 2))

        @test value.(fft(A, 1)) == fft(value.(A), 1)
        @test complexpartials.(fft(A, 1), 1) == fft(partials.(A, 1), 1)
        @test complexpartials.(fft(A, 1), 2) == fft(partials.(A, 2), 1)

        @test value.(fft(A, 2)) == fft(value.(A), 2)
        @test complexpartials.(fft(A, 2), 1) == fft(partials.(A, 1), 2)
        @test complexpartials.(fft(A, 2), 2) == fft(partials.(A, 2), 2)
    end

    @testset "complex" begin
        @test fft(c1) ≈ fft(real(c1)) + im*fft(imag(c1))
        dest = similar(c1)
        @test mul!(dest, plan_fft(x1), x1) == fft(x1) == dest
        @test mul!(dest, plan_fft(c1), c1) == fft(c1) == dest

        C = c1 * ((1:10) .+ im*(2:11))'
        @test fft(C) ≈ fft(real(C)) + im*fft(imag(C))
        dest = similar(C)
        @test mul!(dest, plan_fft(C), C) == fft(C) == dest
    end
end