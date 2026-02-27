module AbstractFFTs

using Base.ScopedValues

export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
       fftdims, fftshift, ifftshift, fftshift!, ifftshift!, Frequencies, fftfreq, rfftfreq

include("definitions.jl")
include("TestUtils.jl")

end # module
