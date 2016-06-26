module MinFinder

    using Optim

    #import Optim: optimize

    export optimize, Fminfinder

    immutable Fminfinder <: Optim.Optimizer
    end

    include("minfinder_main.jl")
    include(joinpath("problems", "multiple_minima.jl"))

end # module
