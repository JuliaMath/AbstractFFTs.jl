module AbstractFFTsForwardDiffExt

using AbstractFFTs
import ForwardDiff
import ForwardDiff: Dual
import AbstractFFTs: Plan

for P in (:Plan, :ScaledPlan)  # need ScaledPlan to avoid ambiguities
    @eval begin
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{DT}) where DT<:Dual = array2dual(DT, p * dual2array(x))
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{Complex{DT}}) where DT<:Dual = array2dual(DT, p * dual2array(x))
    end
end

mul!(y::AbstractArray{<:Union{Dual,Complex{<:Dual}}}, p::Plan, x::AbstractArray{<:Union{Dual,Complex{<:Dual}}}) = copyto!(y, p*x)

AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

dual2array(x::Array{<:Dual{Tag,T}}) where {Tag,T} = reinterpret(reshape, T, x)
dual2array(x::Array{<:Complex{<:Dual{Tag, T}}}) where {Tag,T} = complex.(dual2array(real(x)), dual2array(imag(x)))
array2dual(DT::Type{<:Dual}, x::Array{T}) where T = reinterpret(reshape, DT, real(x))
array2dual(DT::Type{<:Dual}, x::Array{<:Complex{T}}) where T = complex.(array2dual(DT, real(x)), array2dual(DT, imag(x)))


for plan in (:plan_fft, :plan_ifft, :plan_bfft, :plan_rfft)
    @eval begin
        AbstractFFTs.$plan(x::AbstractArray{<:Dual}, dims=1:ndims(x)) = AbstractFFTs.$plan(dual2array(x), 1 .+ dims)
        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:Dual}}, dims=1:ndims(x)) = AbstractFFTs.$plan(dual2array(x), 1 .+ dims)
    end
end



end # module