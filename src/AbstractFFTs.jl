module AbstractFFTs

export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
       fftdims, fftshift, ifftshift, fftshift!, ifftshift!, Frequencies, fftfreq, rfftfreq

include("definitions.jl")

@static if !isdefined(Base, :get_extension)
    import Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        Requires.@require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin
            include("../ext/AbstractFFTsChainRulesCoreExt.jl")
        end
    end
end

end # module
