__precompile__()
module AbstractFFTs

using Compat

# After this version, the bindings can overwrite deprecated bindings in Base safely, but
# prior to it we want to extend/reexport the Base definitions
if VERSION < v"0.7.0-DEV.602"
    import Base: fft, ifft, bfft, fft!, ifft!, bfft!,
                 plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
                 rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
                 fftshift, ifftshift
    import Base.DFT: Plan, ScaledPlan, plan_inv, pinv_type, normalization,
                     rfft_output_size, brfft_output_size, realfloat, complexfloat
end
# Reexport the Base bindings unchanged for versions before FFTW was removed, or export the
# new definitions after overwritable deprecation bindings were introduced
export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
       fftshift, ifftshift

# Only define things if we aren't using the existing Base bindings
VERSION >= v"0.7.0-DEV.602" && include("definitions.jl")

end # module
