module AbstractFFTsChainRulesCoreExt

using AbstractFFTs
import ChainRulesCore

function ChainRulesCore.frule((_, Δx, _), ::typeof(fft), x::AbstractArray, dims)
    y = fft(x, dims)
    Δy = fft(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(fft), x::AbstractArray, dims)
    y = fft(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function fft_pullback(ȳ)
        x̄ = project_x(bfft(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, fft_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(rfft), x::AbstractArray{<:Real}, dims)
    y = rfft(x, dims)
    Δy = rfft(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(rfft), x::AbstractArray{<:Real}, dims)
    y = rfft(x, dims)

    # compute scaling factors
    halfdim = first(dims)
    d = size(x, halfdim)
    n = size(y, halfdim)
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? 1 : 2 for i in 1:n],
        ntuple(i -> i == first(dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function rfft_pullback(ȳ)
        ybar = ChainRulesCore.unthunk(ȳ)
        _scale = convert(typeof(ybar),scale)
        x̄ = project_x(brfft(ybar ./ _scale, d, dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, rfft_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(ifft), x::AbstractArray, dims)
    y = ifft(x, dims)
    Δy = ifft(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(ifft), x::AbstractArray, dims)
    y = ifft(x, dims)
    invN = AbstractFFTs.normalization(y, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function ifft_pullback(ȳ)
        x̄ = project_x(invN .* fft(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, ifft_pullback
end

function ChainRulesCore.frule((_, Δx, _, _), ::typeof(irfft), x::AbstractArray, d::Int, dims)
    y = irfft(x, d, dims)
    Δy = irfft(Δx, d, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(irfft), x::AbstractArray, d::Int, dims)
    y = irfft(x, d, dims)

    # compute scaling factors
    halfdim = first(dims)
    n = size(x, halfdim)
    invN = AbstractFFTs.normalization(y, dims)
    twoinvN = 2 * invN
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? invN : twoinvN for i in 1:n],
        ntuple(i -> i == first(dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function irfft_pullback(ȳ)
        ybar = ChainRulesCore.unthunk(ȳ)
        _scale = convert(typeof(ybar),scale)
        x̄ = project_x(_scale .* rfft(real.(ybar), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, irfft_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(bfft), x::AbstractArray, dims)
    y = bfft(x, dims)
    Δy = bfft(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(bfft), x::AbstractArray, dims)
    y = bfft(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function bfft_pullback(ȳ)
        x̄ = project_x(fft(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, bfft_pullback
end

function ChainRulesCore.frule((_, Δx, _, _), ::typeof(brfft), x::AbstractArray, d::Int, dims)
    y = brfft(x, d, dims)
    Δy = brfft(Δx, d, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(brfft), x::AbstractArray, d::Int, dims)
    y = brfft(x, d, dims)

    # compute scaling factors
    halfdim = first(dims)
    n = size(x, halfdim)
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? 1 : 2 for i in 1:n],
        ntuple(i -> i == first(dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function brfft_pullback(ȳ)
        x̄ = project_x(scale .* rfft(real.(ChainRulesCore.unthunk(ȳ)), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, brfft_pullback
end

# shift functions
function ChainRulesCore.frule((_, Δx, _), ::typeof(fftshift), x::AbstractArray, dims)
    y = fftshift(x, dims)
    Δy = fftshift(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(fftshift), x::AbstractArray, dims)
    y = fftshift(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function fftshift_pullback(ȳ)
        x̄ = project_x(ifftshift(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, fftshift_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(ifftshift), x::AbstractArray, dims)
    y = ifftshift(x, dims)
    Δy = ifftshift(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(ifftshift), x::AbstractArray, dims)
    y = ifftshift(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function ifftshift_pullback(ȳ)
        x̄ = project_x(fftshift(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, ifftshift_pullback
end

# plans
function ChainRulesCore.frule((_, _, Δx), ::typeof(*), P::AbstractFFTs.Plan, x::AbstractArray) 
    y = P * x
    if Base.mightalias(y, x)
        throw(ArgumentError("differentiation rules are not supported for in-place plans"))
    end
    Δy = P * Δx
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), P::AbstractFFTs.Plan, x::AbstractArray)
    y = P * x
    if Base.mightalias(y, x)
        throw(ArgumentError("differentiation rules are not supported for in-place plans"))
    end
    project_x = ChainRulesCore.ProjectTo(x)
    Pt = P'
    function mul_plan_pullback(ȳ)
        x̄ = project_x(Pt * ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
    end
    return y, mul_plan_pullback
end

function ChainRulesCore.frule((_, ΔP, Δx), ::typeof(*), P::AbstractFFTs.ScaledPlan, x::AbstractArray) 
    y = P * x 
    if Base.mightalias(y, x)
        throw(ArgumentError("differentiation rules are not supported for in-place plans"))
    end
    Δy = P * Δx .+ (ΔP.scale / P.scale) .* y
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), P::AbstractFFTs.ScaledPlan, x::AbstractArray)
    y = P * x
    if Base.mightalias(y, x)
        throw(ArgumentError("differentiation rules are not supported for in-place plans"))
    end
    Pt = P'
    scale = P.scale
    project_x = ChainRulesCore.ProjectTo(x)
    project_scale = ChainRulesCore.ProjectTo(scale)
    function mul_scaledplan_pullback(ȳ)
        x̄ = ChainRulesCore.@thunk(project_x(Pt * ȳ))
        scale_tangent = ChainRulesCore.@thunk(project_scale(AbstractFFTs.dot(y, ȳ) / conj(scale)))
        plan_tangent = ChainRulesCore.Tangent{typeof(P)}(;p=ChainRulesCore.NoTangent(), scale=scale_tangent)
        return ChainRulesCore.NoTangent(), plan_tangent, x̄ 
    end
    return y, mul_scaledplan_pullback
end

end # module

