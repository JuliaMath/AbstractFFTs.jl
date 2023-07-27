module TestPlans

using LinearAlgebra
using AbstractFFTs
using AbstractFFTs: Plan

mutable struct TestPlan{T,N,G} <: Plan{T}
    region::G
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function TestPlan{T}(region::G, sz::NTuple{N,Int}) where {T,N,G}
        return new{T,N,G}(region, sz)
    end
end

mutable struct InverseTestPlan{T,N,G} <: Plan{T}
    region::G
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function InverseTestPlan{T}(region::G, sz::NTuple{N,Int}) where {T,N,G}
        return new{T,N,G}(region, sz)
    end
end

Base.size(p::TestPlan) = p.sz
Base.ndims(::TestPlan{T,N}) where {T,N} = N
Base.size(p::InverseTestPlan) = p.sz
Base.ndims(::InverseTestPlan{T,N}) where {T,N} = N

AbstractFFTs.AdjointStyle(::TestPlan) = AbstractFFTs.FFTAdjointStyle()
AbstractFFTs.AdjointStyle(::InverseTestPlan) = AbstractFFTs.FFTAdjointStyle()

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...) where {T}
    return TestPlan{T}(region, size(x))
end
function AbstractFFTs.plan_bfft(x::AbstractArray{T}, region; kwargs...) where {T}
    return InverseTestPlan{T}(region, size(x))
end

function AbstractFFTs.plan_inv(p::TestPlan{T}) where {T}
    unscaled_pinv = InverseTestPlan{T}(p.region, p.sz)
    N = AbstractFFTs.normalization(T, p.sz, p.region)
    unscaled_pinv.pinv = AbstractFFTs.ScaledPlan(p, N)
    pinv = AbstractFFTs.ScaledPlan(unscaled_pinv, N)
    return pinv
end
function AbstractFFTs.plan_inv(pinv::InverseTestPlan{T}) where {T}
    unscaled_p = TestPlan{T}(pinv.region, pinv.sz)
    N = AbstractFFTs.normalization(T, pinv.sz, pinv.region)
    unscaled_p.pinv = AbstractFFTs.ScaledPlan(pinv, N)
    p = AbstractFFTs.ScaledPlan(unscaled_p, N)
    return p
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

function LinearAlgebra.mul!(
    y::AbstractArray{<:Complex,N}, p::TestPlan, x::AbstractArray{<:Union{Complex,Real},N}
) where {N}
    size(y) == size(p) == size(x) || throw(DimensionMismatch())
    dft!(y, x, p.region, -1)
end
function LinearAlgebra.mul!(
    y::AbstractArray{<:Complex,N}, p::InverseTestPlan, x::AbstractArray{<:Union{Complex,Real},N}
) where {N}
    size(y) == size(p) == size(x) || throw(DimensionMismatch())
    dft!(y, x, p.region, 1)
end

Base.:*(p::TestPlan, x::AbstractArray) = mul!(similar(x, complex(float(eltype(x)))), p, x)
Base.:*(p::InverseTestPlan, x::AbstractArray) = mul!(similar(x, complex(float(eltype(x)))), p, x)

mutable struct TestRPlan{T,N,G} <: Plan{T}
    region::G
    sz::NTuple{N,Int}
    pinv::Plan{Complex{T}}
    TestRPlan{T}(region::G, sz::NTuple{N,Int}) where {T,N,G} = new{T,N,G}(region, sz)
end

mutable struct InverseTestRPlan{T,N,G} <: Plan{Complex{T}}
    d::Int
    region::G
    sz::NTuple{N,Int}
    pinv::Plan{T}
    function InverseTestRPlan{T}(d::Int, region::G, sz::NTuple{N,Int}) where {T,N,G}
        sz[first(region)::Int] == d ÷ 2 + 1 || error("incompatible dimensions")
        return new{T,N,G}(d, region, sz)
    end
end

AbstractFFTs.AdjointStyle(::TestRPlan) = AbstractFFTs.RFFTAdjointStyle()
AbstractFFTs.AdjointStyle(p::InverseTestRPlan) = AbstractFFTs.IRFFTAdjointStyle(p.d)

function AbstractFFTs.plan_rfft(x::AbstractArray{T}, region; kwargs...) where {T<:Real}
    return TestRPlan{T}(region, size(x))
end
function AbstractFFTs.plan_brfft(x::AbstractArray{Complex{T}}, d, region; kwargs...) where {T}
    return InverseTestRPlan{T}(d, region, size(x))
end
function AbstractFFTs.plan_inv(p::TestRPlan{T,N}) where {T,N}
    firstdim = first(p.region)::Int
    d = p.sz[firstdim]
    sz = ntuple(i -> i == firstdim ? d ÷ 2 + 1 : p.sz[i], Val(N))
    _N = AbstractFFTs.normalization(T, p.sz, p.region)

    unscaled_pinv = InverseTestRPlan{T}(d, p.region, sz)
    unscaled_pinv.pinv = AbstractFFTs.ScaledPlan(p, _N)
    pinv = AbstractFFTs.ScaledPlan(unscaled_pinv, _N)
    return pinv
end

function AbstractFFTs.plan_inv(pinv::InverseTestRPlan{T,N}) where {T,N}
    firstdim = first(pinv.region)::Int
    sz = ntuple(i -> i == firstdim ? pinv.d : pinv.sz[i], Val(N))
    _N = AbstractFFTs.normalization(T, sz, pinv.region)

    unscaled_p = TestRPlan{T}(pinv.region, sz)
    unscaled_p.pinv = AbstractFFTs.ScaledPlan(pinv, _N)
    p = AbstractFFTs.ScaledPlan(unscaled_p, _N)
    return p
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

function LinearAlgebra.mul!(y::AbstractArray{<:Complex, N}, p::TestRPlan, x::AbstractArray{<:Real, N}) where {N}
    size(p) == size(x) || error("array and plan are not consistent")

    # compute DFT
    dft!(y, x, p.region, -1)

    # we clean the output a bit to make sure that we return real values
    # whenever the output is mathematically guaranteed to be a real number
    firstdim = first(p.region)::Int
    d = size(x, firstdim)
    firstdim_size = d ÷ 2 + 1
    to_real!(selectdim(y, firstdim, 1))
    if iseven(d)
        to_real!(selectdim(y, firstdim, firstdim_size))
    end

    return y
end

function Base.:*(p::TestRPlan, x::AbstractArray)
    # create output array
    firstdim = first(p.region)::Int
    d = size(x, firstdim)
    firstdim_size = d ÷ 2 + 1
    T = complex(float(eltype(x)))
    sz = ntuple(i -> i == firstdim ? firstdim_size : size(x, i), Val(ndims(x)))
    y = similar(x, T, sz)

    # run in-place mul!
    mul!(y, p, x)

    return y
end

function LinearAlgebra.mul!(y::AbstractArray{<:Real, N}, p::InverseTestRPlan, x::AbstractArray{<:Complex, N}) where {N}
    size(p) == size(x) || error("array and plan are not consistent")

    # compute DFT
    real_invdft!(y, x, p.region)
end

function Base.:*(p::InverseTestRPlan, x::AbstractArray)
    # create output array
    firstdim = first(p.region)::Int
    d = p.d
    sz = ntuple(i -> i == firstdim ? d : size(x, i), Val(ndims(x)))
    y = similar(x, real(float(eltype(x))), sz)

    # run in-place mul!
    mul!(y, p, x)

    return y
end

# In-place plans
# (simple wrapper of OOP plans)
struct InplaceTestPlan{T,P<:Plan{T}} <: Plan{T}
    plan::P
end

Base.size(p::InplaceTestPlan) = size(p.plan)
Base.ndims(p::InplaceTestPlan) = ndims(p.plan)
AbstractFFTs.fftdims(p::InplaceTestPlan) = fftdims(p.plan)
AbstractFFTs.AdjointStyle(p::InplaceTestPlan) = AbstractFFTs.AdjointStyle(p.plan)

function AbstractFFTs.plan_fft!(x::AbstractArray, region; kwargs...)
    return InplaceTestPlan(plan_fft(x, region; kwargs...))
end
function AbstractFFTs.plan_bfft!(x::AbstractArray, region; kwargs...)
    return InplaceTestPlan(plan_bfft(x, region; kwargs...))
end

Base.:*(p::InplaceTestPlan, x::AbstractArray) = copyto!(x, p.plan * x)

AbstractFFTs.plan_inv(p::InplaceTestPlan) = InplaceTestPlan(AbstractFFTs.plan_inv(p.plan))
# Don't cache inverse of inplace wrapper plan (only inverse of inner plan)
Base.inv(p::InplaceTestPlan) = InplaceTestPlan(inv(p.plan))

end