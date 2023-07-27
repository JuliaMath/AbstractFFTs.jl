# Public Interface

## FFT and FFT planning functions

```@docs
AbstractFFTs.fft
AbstractFFTs.fft!
AbstractFFTs.ifft
AbstractFFTs.ifft!
AbstractFFTs.bfft
AbstractFFTs.bfft!
AbstractFFTs.plan_fft
AbstractFFTs.plan_ifft
AbstractFFTs.plan_bfft
AbstractFFTs.plan_fft!
AbstractFFTs.plan_ifft!
AbstractFFTs.plan_bfft!
AbstractFFTs.rfft
AbstractFFTs.irfft
AbstractFFTs.brfft
AbstractFFTs.plan_rfft
AbstractFFTs.plan_brfft
AbstractFFTs.plan_irfft
AbstractFFTs.fftdims
AbstractFFTs.fftshift
AbstractFFTs.fftshift!
AbstractFFTs.ifftshift
AbstractFFTs.ifftshift!
AbstractFFTs.fftfreq
AbstractFFTs.rfftfreq
Base.size
```

## Adjoint functionality

The following API is supported by plans that support adjoint functionality.
It is also relevant to implementers of FFT plans that wish to support adjoints.
```@docs
Base.adjoint
AbstractFFTs.AdjointStyle
AbstractFFTs.output_size
AbstractFFTs.adjoint_mul
AbstractFFTs.FFTAdjointStyle
AbstractFFTs.RFFTAdjointStyle
AbstractFFTs.IRFFTAdjointStyle
AbstractFFTs.UnitaryAdjointStyle
```
