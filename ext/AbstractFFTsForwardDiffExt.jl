module AbstractFFTsForwardDiffExt

using AbstractFFTs
using AbstractFFTs.LinearAlgebra
import ForwardDiff
import ForwardDiff: Dual
import AbstractFFTs: Plan, mul!, dualplan, dual2array


AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

dual2array(x::Array{<:Dual{Tag,T}}) where {Tag,T} = reinterpret(reshape, T, x)
dual2array(x::Array{<:Complex{<:Dual{Tag, T}}}) where {Tag,T} = complex.(dual2array(real(x)), dual2array(imag(x)))
array2dual(DT::Type{<:Dual}, x::Array{T}) where T = reinterpret(reshape, DT, real(x))
array2dual(DT::Type{<:Dual}, x::Array{<:Complex{T}}) where T = complex.(array2dual(DT, real(x)), array2dual(DT, imag(x)))


########
# DualPlan
# represents a plan acting on dual numbers. We wrap a plan acting on a higher dimensional tensor
# as an array of duals can be reinterpreted as a higher dimensional array.
# This allows standard FFTW plans to act on arrays of duals.
#####
struct DualPlan{T,P} <: Plan{T}
    p::P
    DualPlan{T,P}(p) where {T,P} = new(p)
end

DualPlan(::Type{Dual{Tag,V,N}}, p::Plan{T}) where {Tag,T<:Real,V,N} = DualPlan{Dual{Tag,T,N},typeof(p)}(p)
DualPlan(::Type{Dual{Tag,V,N}}, p::Plan{Complex{T}}) where {Tag,T<:Real,V,N} = DualPlan{Complex{Dual{Tag,T,N}},typeof(p)}(p)
dualplan(D, p) = DualPlan(D, p)
Base.size(p::DualPlan) = Base.tail(size(p.p))
Base.:*(p::DualPlan{DT}, x::AbstractArray{DT}) where DT<:Dual = array2dual(DT, p.p * dual2array(x))
Base.:*(p::DualPlan{Complex{DT}}, x::AbstractArray{Complex{DT}}) where DT<:Dual = array2dual(DT, p.p * dual2array(x))

function LinearAlgebra.mul!(y::AbstractArray{<:Dual}, p::DualPlan, x::AbstractArray{<:Dual})
    LinearAlgebra.mul!(dual2array(y), p.p, dual2array(x)) # even though `Dual` are immutable, when in an `Array` they can be modified.
    y
end

function LinearAlgebra.mul!(y::AbstractArray{<:Complex{<:Dual}}, p::DualPlan, x::AbstractArray{<:Union{Dual,Complex{<:Dual}}})
    copyto!(y, p*x) # Complex duals cannot be reinterpret in-place
end


for plan in (:plan_fft, :plan_ifft, :plan_bfft, :plan_rfft)
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{D}, dims=1:ndims(x)) where D<:Dual = dualplan(D, AbstractFFTs.$plan(dual2array(x), 1 .+ dims))
        AbstractFFTs.$plan(x::AbstractArray{<:Complex{D}}, dims=1:ndims(x)) where D<:Dual = dualplan(D, AbstractFFTs.$plan(dual2array(x), 1 .+ dims))
    end
end


for plan in (:plan_irfft, :plan_brfft)  # these take an extra argument, only when complex?
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{D}, dims=1:ndims(x)) where D<:Dual = dualplan(D, AbstractFFTs.$plan(dual2array(x), 1 .+ dims))
        AbstractFFTs.$plan(x::AbstractArray{<:Complex{D}}, d::Integer, mdims=1:ndims(x)) where D<:Dual = dualplan(D, AbstractFFTs.$plan(dual2array(x), d, 1 .+ dims))
    end
end


end # module