using AbstractFFTs
using AbstractFFTs: Plan
using ChainRulesTestUtils

using LinearAlgebra
using Random
using Test

import Unitful

Random.seed!(1234)

const GROUP = get(ENV, "GROUP", "All")

include("TestPlans.jl")
include("testfft.jl")

if GROUP == "All" || GROUP == "TestPlans"
    using .TestPlans
    testfft()
elseif GROUP == "All" || GROUP == "FFTW" # integration test with FFTW
    using FFTW
    testfft()
end

