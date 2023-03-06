# ffts
# we explicitly handle both unprovided and provided dims arguments in all rules, which
# results in some additional complexity here but means no assumptions are made on what
# signatures downstream implementations support.
function ChainRulesCore.frule(Δargs, ::typeof(fft), x::AbstractArray, dims=nothing)
    Δx =  Δargs[2]
    dims_args = (dims === nothing) ? () : (dims,)
    y = fft(x, dims_args...)
    Δy = fft(Δx, dims_args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(fft), x::AbstractArray, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    y = fft(x, dims_args...)
    project_x = ChainRulesCore.ProjectTo(x)
    function fft_pullback(ȳ)
        x̄ = project_x(bfft(ChainRulesCore.unthunk(ȳ), dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, dims_args_tangent... 
    end
    return y, fft_pullback
end

function ChainRulesCore.frule(Δargs, ::typeof(rfft), x::AbstractArray{<:Real}, dims=nothing)
    Δx = Δargs[2]
    dims_args = (dims === nothing) ? () : (dims,)
    y = rfft(x, dims_args...)
    Δy = rfft(Δx, dims_args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(rfft), x::AbstractArray{<:Real}, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    true_dims = (dims === nothing) ? (1:ndims(x)) : dims 
    y = rfft(x, dims_args...)

    # compute scaling factors
    halfdim = first(true_dims)
    d = size(x, halfdim)
    n = size(y, halfdim)
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? 1 : 2 for i in 1:n],
        ntuple(i -> i == first(true_dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function rfft_pullback(ȳ)
        x̄ = project_x(brfft(ChainRulesCore.unthunk(ȳ) ./ scale, d, dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, dims_args_tangent... 
    end
    return y, rfft_pullback
end

function ChainRulesCore.frule(Δargs, ::typeof(ifft), x::AbstractArray, dims=nothing)
    Δx = Δargs[2]
    args = (dims === nothing) ? () : (dims,)
    y = ifft(x, args...)
    Δy = ifft(Δx, args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(ifft), x::AbstractArray, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    true_dims = (dims === nothing) ? (1:ndims(x)) : dims 
    y = ifft(x, dims_args...)
    invN = normalization(y, true_dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function ifft_pullback(ȳ)
        x̄ = project_x(invN .* fft(ChainRulesCore.unthunk(ȳ), dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, dims_args_tangent...
    end
    return y, ifft_pullback
end

function ChainRulesCore.frule(Δargs, ::typeof(irfft), x::AbstractArray, d::Int, dims=nothing)
    Δx = Δargs[2]
    dims_args = (dims === nothing) ? () : (dims,)
    y = irfft(x, d, dims_args...)
    Δy = irfft(Δx, d, dims_args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(irfft), x::AbstractArray, d::Int, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    true_dims = (dims === nothing) ? (1:ndims(x)) : dims 
    y = irfft(x, d, dims_args...)

    # compute scaling factors
    halfdim = first(true_dims)
    n = size(x, halfdim)
    invN = normalization(y, true_dims)
    twoinvN = 2 * invN
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? invN : twoinvN for i in 1:n],
        ntuple(i -> i == first(true_dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function irfft_pullback(ȳ)
        x̄ = project_x(scale .* rfft(real.(ChainRulesCore.unthunk(ȳ)), dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), dims_args_tangent...
    end
    return y, irfft_pullback
end

function ChainRulesCore.frule(Δargs, ::typeof(bfft), x::AbstractArray, dims=nothing)
    Δx = Δargs[2]
    dims_args = (dims === nothing) ? () : (dims,)
    y = bfft(x, dims_args...)
    Δy = bfft(Δx, dims_args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(bfft), x::AbstractArray, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    y = bfft(x, dims_args...)
    project_x = ChainRulesCore.ProjectTo(x)
    function bfft_pullback(ȳ)
        x̄ = project_x(fft(ChainRulesCore.unthunk(ȳ), dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, dims_args_tangent...
    end
    return y, bfft_pullback
end

function ChainRulesCore.frule(Δargs, ::typeof(brfft), x::AbstractArray, d::Int, dims=nothing)
    Δx = Δargs[2]
    dims_args = (dims === nothing) ? () : (dims,)
    y = brfft(x, d, dims_args...)
    Δy = brfft(Δx, d, dims_args...)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(brfft), x::AbstractArray, d::Int, dims=nothing)
    dims_args = (dims === nothing) ? () : (dims,)
    true_dims = (dims === nothing) ? (1:ndims(x)) : dims 
    y = brfft(x, d, dims_args...)

    # compute scaling factors
    halfdim = first(true_dims)
    n = size(x, halfdim)
    scale = reshape(
        [i == 1 || (i == n && 2 * (i - 1) == d) ? 1 : 2 for i in 1:n],
        ntuple(i -> i == first(true_dims) ? n : 1, Val(ndims(x))),
    )

    project_x = ChainRulesCore.ProjectTo(x)
    function brfft_pullback(ȳ)
        x̄ = project_x(scale .* rfft(real.(ChainRulesCore.unthunk(ȳ)), dims_args...))
        dims_args_tangent = (dims === nothing) ? () : (ChainRulesCore.NoTangent(),)
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), dims_args_tangent...
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
