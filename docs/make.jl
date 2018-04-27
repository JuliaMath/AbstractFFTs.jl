using Documenter, AbstractFFTs

makedocs(
    modules = [AbstractFFTs],
    clean = false,
    format = :html,
    sitename = "AbstractFFTs.jl",
    pages = Any[
        "Home" => "index.md",
        "API" => "api.md",
        "Implementations" => "implementations.md",
    ],
)

deploydocs(
    julia = "nightly",
    repo = "github.com/JuliaMath/AbstractFFTs.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
