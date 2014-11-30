module MinFinder

    using Optim

    export minfinder

    include("minfinder_main.jl")

    include(joinpath("problems", "multiple_minima.jl"))

end # module
