__precompile__()
module AbstractFFTs

# After this version, the bindings can overwrite deprecated bindings in Base safely, but
# prior to it we want to extend/reexport the Base definitions
if VERSION < v"0.7.0-DEV.986"
    import Base: fft, ifft, bfft, fft!, ifft!, bfft!,
                 plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
                 rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
                 fftshift, ifftshift
    if VERSION < v"0.7.0-DEV.602"
        import Base.DFT: Plan, plan_inv
    end
end
# Reexport the Base bindings unchanged for versions before FFTW was removed, or export the
# new definitions after overwritable deprecation bindings were introduced
if VERSION < v"0.7.0-DEV.602" || VERSION >= v"0.7.0-DEV.986"
    export fft, ifft, bfft, fft!, ifft!, bfft!,
           plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
           rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
           fftshift, ifftshift
end

# Only define things if we aren't using the existing Base bindings
VERSION >= v"0.7.0-DEV.602" && include("definitions.jl")

end # module
