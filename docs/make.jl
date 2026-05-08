using Documenter
using RegisterFit
using RegisterCore

DocMeta.setdocmeta!(RegisterFit, :DocTestSetup, :(using RegisterFit, RegisterCore); recursive=true)

makedocs(;
    modules=[RegisterFit],
    sitename="RegisterFit.jl",
    format=Documenter.HTML(;
        canonical="https://HolyLab.github.io/RegisterFit.jl",
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/RegisterFit.jl",
    devbranch="master",
)
