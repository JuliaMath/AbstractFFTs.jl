var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "DocTestSetup = :(using AbstractFFTs)\nCurrentModule = AbstractFFTs"
},

{
    "location": "#AbstractFFTs.jl-1",
    "page": "Home",
    "title": "AbstractFFTs.jl",
    "category": "section",
    "text": "This package provides a generic framework for defining fast Fourier transform (FFT) implementations in Julia. The code herein was part of Julia\'s Base library for Julia versions 0.6 and lower."
},

{
    "location": "#Contents-1",
    "page": "Home",
    "title": "Contents",
    "category": "section",
    "text": "Pages = [\"api.md\", \"implementations.md\"]"
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#Public-Interface-1",
    "page": "API",
    "title": "Public Interface",
    "category": "section",
    "text": "AbstractFFTs.fft\nAbstractFFTs.fft!\nAbstractFFTs.ifft\nAbstractFFTs.ifft!\nAbstractFFTs.bfft\nAbstractFFTs.bfft!\nAbstractFFTs.plan_fft\nAbstractFFTs.plan_ifft\nAbstractFFTs.plan_bfft\nAbstractFFTs.plan_fft!\nAbstractFFTs.plan_ifft!\nAbstractFFTs.plan_bfft!\nAbstractFFTs.rfft\nAbstractFFTs.irfft\nAbstractFFTs.brfft\nAbstractFFTs.plan_rfft\nAbstractFFTs.plan_brfft\nAbstractFFTs.plan_irfft\nAbstractFFTs.fftshift(::Any)\nAbstractFFTs.fftshift(::Any, ::Any)\nAbstractFFTs.ifftshift"
},

{
    "location": "implementations/#",
    "page": "Implementations",
    "title": "Implementations",
    "category": "page",
    "text": ""
},

{
    "location": "implementations/#FFT-Implementations-1",
    "page": "Implementations",
    "title": "FFT Implementations",
    "category": "section",
    "text": ""
},

{
    "location": "implementations/#Existing-packages-1",
    "page": "Implementations",
    "title": "Existing packages",
    "category": "section",
    "text": "The following packages extend the functionality provided by AbstractFFTs:FFTW.jl: Bindings for the FFTW library. This also used to be part of Base Julia."
},

{
    "location": "implementations/#Defining-a-new-implementation-1",
    "page": "Implementations",
    "title": "Defining a new implementation",
    "category": "section",
    "text": "Implementations should implement LinearAlgebra.mul!(Y, plan, X) (or A_mul_B!(y, p::MyPlan, x) on Julia prior to 0.7.0-DEV.3204) so as to support pre-allocated output arrays. We don\'t define * in terms of mul! generically here, however, because of subtleties for in-place and real FFT plans.To support inv, \\, and ldiv!(y, plan, x), we require Plan subtypes to have a pinv::Plan field, which caches the inverse plan, and which should be initially undefined. They should also implement plan_inv(p) to construct the inverse of a plan p.Implementations only need to provide the unnormalized backwards FFT, similar to FFTW, and we do the scaling generically to get the inverse FFT."
},

]}
