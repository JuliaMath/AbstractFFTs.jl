using Documenter, AbstractFFTs

makedocs(
    modules = [AbstractFFTs],
    sitename = "AbstractFFTs.jl",
    pages = Any[
        "Home" => "index.md",
        "API" => "api.md",
        "Implementations" => "implementations.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaMath/AbstractFFTs.jl.git",
    target = "build",
    push_preview = true
)
