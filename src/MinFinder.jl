module MinFinder

    using Optim
    using OptionsMod

    export minfinder

    include("minfinder_main.jl")

    include(joinpath("problems", "multiple_minima.jl"))

end # module
